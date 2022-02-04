import numpy as np
from pathlib import Path
import configparser
import platform
from subprocess import PIPE, CalledProcessError, run, check_output, list2cmdline
import sys
from functools import wraps
import inspect
from pytest_easyMPI import mpi_parallel


class ExampleChecker:
	""" Used to check example output from pytest tests """

	def __init__(self, input_file):
		self._input_file = input_file
		self._example_path = input_file.parent
		self._config = configparser.ConfigParser(inline_comment_prefixes='#')
		self._config.optionxform = str
		self._config.read(str(input_file))
		self._outfile = self._config.get("OUTFILE", "potential", fallback="fitsnap_potential")
		self._standard_path = self._check_for_standard()

	def _check_for_standard(self):
		""" Determine directory that contains the output standard """
		possibilites = self._example_path.glob('*_Standard')
		for possibility in possibilites:
			if possibility.is_dir():
				return possibility

	def snapcoeff_diff(self):
		""" Check if current output differs too much from standardized output"""
		self._outfile += ".snapcoeff"
		standard_outfile = str(self._standard_path / Path(self._outfile).name)
		testing_coeffs = self._snap_parser(str(self._example_path / self._outfile))
		standard_coeffs = self._snap_parser(standard_outfile)
		assert np.max(testing_coeffs - standard_coeffs) < 1e-6

	@staticmethod
	def _snap_parser(coeff_file):
		""" Parse SNAP coeff files """
		coeffs = []
		with open(coeff_file, 'r') as readin:
			lines = readin.readlines()
			ndescs = int(lines[2].split()[-1])
			for descind in range(ndescs):
				line = lines[4 + descind].split()
				# l = line.split()
				# descnew.append(line.split('#')[-1])
				coeffs.append(float(line[0]))
		return np.array(coeffs)


def assert_mpi(method):
	""" Decorator to be used in MPICheck to assert methods fail with stubs on """
	def stub_function(*args, **kw):
		assert args[0].__class__.__name__ == 'MPICheck'
		assert args[0].stubs == 0
		return method(*args, **kw)
	return stub_function


class MPICheck:
	""" Container object for mpi related namespace and tools """
	counter = 0

	def __init__(self):
		MPICheck.counter += 1
		self.system = platform.system()

		try:
			# stubs = 0 MPI is active
			self.stubs = 0
			from mpi4py import MPI
			self._MPI = MPI
			self.comm = MPI.COMM_WORLD
			self.rank = self.comm.Get_rank()
			self.size = self.comm.Get_size()
			self._mpi_shared_lib = MPI.__file__
			self._mpirun, self._mpiexec, self._orterun = self._find_mpirun()
		except ModuleNotFoundError:
			# stubs = 1 MPI not active
			self.stubs = 1
			self.rank = 0
			self.size = 1
			self.comm = None
			self.sub_rank = 0
			self.sub_size = 1
			self.sub_comm = None
			self.sub_head_proc = 0
			self.node_index = 0
			self.number_of_nodes = 1

	def finalize(self):
		self._MPI.Finalize()

	@assert_mpi
	def get_mpi_executable(self, exec_type="mpirun"):
		""" Getter for MPI executable default to mpirun """
		executable = None
		if exec_type == "mpirun":
			executable = self._mpirun
		if exec_type == "mpiexec" or executable is None:
			executable = self._mpiexec
		if exec_type == "orterun" or executable is None:
			executable = self._orterun
		assert executable is not None
		return executable

	@assert_mpi
	def set_added_comm(self):
		""" Assign ADDED COMM to current comm, which inserts current comm into future instances of FitSNAP """
		self._MPI.ADDED_COMM = self.comm

	@assert_mpi
	def _find_mpirun(self):
		""" Find mpirun, mpiexec, or orterun"""
		mpilib = Path(self._dylib_reader())
		mpibin = None
	#	for i in range(1,len(mpilib.parts)):
	#		if mpilib.parts[-i]=='lib':
	#			mpibin = mpilib.parents[i-1] / 'bin' 
	#			break
		if mpibin is None:
			mpibin = mpilib.parent.parent / 'bin'
		mpirun = str(mpibin / 'mpirun') if (mpibin / 'mpirun').exists() else None
		mpiexec = str(mpibin / 'mpiexec') if (mpibin / 'mpiexec').exists() else None
		orterun = str(mpibin / 'orterun') if (mpibin / 'orterun').exists() else None
		if mpirun is None and mpiexec is None and orterun is None:
			mpirun = check_output(['which', 'mpirun'], universal_newlines=True).strip()
		print(mpirun)
		# assert Path(mpirun).exists()
		assert mpirun is not None or mpiexec is not None or orterun is not None
		return mpirun, mpiexec, orterun

	def _dylib_reader(self):
		""" Find dynamically linked libraries used by mpi4py """
		if self.system == 'Linux':
			result = run(['ldd', '{}'.format(self._mpi_shared_lib)], stdout=PIPE)
			result = result.stdout.decode('utf-8').replace('\t', '').split('\n')[1:]
			return [string for string in result if 'libmpi' in string][0].split()[2]	
		elif self.system == 'Darwin':
			result = run(['otool', '-L', '{}'.format(self._mpi_shared_lib)], stdout=PIPE)
			result = result.stdout.decode('utf-8').replace('\t', '').split('\n')[1:]
			return [string for string in result if 'libmpi' in string][0].split()[0]
		else:
			raise NotImplementedError('MPI4PY testing only implemented for Linux and MacOS not {}'.format(self.system))


def get_pytest_input(test_func):
	"""
	Convert function object to pytest input string.
	Pytest allows the user to pass the name of an individual
	test to run. This function takes a function object and converts
	it into a string that can be passed to pytest to run only that
	specific test function.
	Arguments
	---------
	test_func : function
		The test function object for which to generate a string input
	Returns
	-------
	str
		Input string for pytest to run the given test function
	"""
	module = inspect.getmodule(test_func).__name__

	module = "/".join(module.split(".")) + ".py"
	func = "::".join(test_func.__qualname__.split("."))

	return f"{module}::{func}"


def mpi_run(nprocs, nnodes=None):
	def decorator(test_func):
		@wraps(test_func)
		def wrapper(*args, **kwargs):
			"""MPI wrap for pytest functions"""
			mpi = MPICheck()
			if mpi.size == 1:
				try:
					the_func = get_pytest_input(test_func).split('::')[1]
					executable = mpi.get_mpi_executable()
					mpi.finalize()
					process = run(
						[
							executable,
							"-np",
							str(nprocs),
							"-oversubscribe",
							sys.executable,
							"-c",
							"from test_examples import {0}; {0}()".format(the_func)
						],
						universal_newlines=True,
						capture_output=True
					)
					output = process.stdout
					print(process.stdout, process.stderr, list2cmdline(process.args))
					if process.returncode:
						raise CalledProcessError(process.returncode, process.args, process.stdout, process.stderr)
					if 'Trouble reading input, exiting...' in output:
						raise RuntimeError('Trouble reading input, exiting...')
					with open("completed_process", "w") as fp:
						print(output, file=fp)
				except CalledProcessError as error:
					with open("failed_process", "w") as fp:
						print(error, file=fp)
					raise RuntimeError("Pytest Failed", error, error.output, error.stderr)
			else:
				test_func(*args, **kwargs)

		return wrapper

	return decorator
