import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import ClassVar

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# Parse command-line early to check for --no-cuda flag
# This needs to happen before get_extensions() is called
# Usage: python setup.py install --no-cuda
#    or: pip install . --no-cuda
BUILD_NO_CUDA = False
if "--no-cuda" in sys.argv:
    BUILD_NO_CUDA = True
    sys.argv.remove("--no-cuda")  # Remove so setuptools doesn't complain
    print("\n" + "=" * 80)
    print("Building CPU-only variant (--no-cuda flag)")
    print("CUDA backend excluded - only eager, triton backends")
    print("=" * 80 + "\n")


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = ""):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir) if source_dir else ""


def _resolve_executable(value: str) -> pathlib.Path | None:
    candidate = pathlib.Path(value).expanduser()
    if candidate.parent != pathlib.Path("."):
        resolved = candidate.resolve()
        return resolved if resolved.is_file() and os.access(resolved, os.X_OK) else None

    located = shutil.which(value)
    return pathlib.Path(located).resolve() if located else None


def _probe_cuda_host_compiler(
    nvcc_bin: pathlib.Path,
    compiler: pathlib.Path,
) -> tuple[bool, str]:
    probe_env = os.environ.copy()
    probe_env.pop("CUDAHOSTCXX", None)
    probe_env.pop("NVCC_CCBIN", None)
    result = subprocess.run(
        [
            str(nvcc_bin),
            "-ccbin",
            str(compiler),
            "-E",
            "-x",
            "cu",
            os.devnull,
            "-o",
            os.devnull,
        ],
        check=False,
        capture_output=True,
        env=probe_env,
        text=True,
    )
    diagnostic_lines = (result.stderr or result.stdout).strip().splitlines()
    diagnostic = next(
        (line.strip() for line in diagnostic_lines if "error:" in line.lower()),
        diagnostic_lines[-1].strip() if diagnostic_lines else "no diagnostic",
    )
    return result.returncode == 0, diagnostic


def get_cuda_host_compiler(nvcc_bin: pathlib.Path) -> pathlib.Path | None:
    """Select an NVCC-compatible C++ compiler without changing the system toolchain."""
    if os.name == "nt":
        return None

    explicit_variables = (
        "COMFY_KITCHEN_CUDA_HOST_COMPILER",
        "NVCC_CCBIN",
        "CUDAHOSTCXX",
        "CXX",
    )
    for variable in explicit_variables:
        value = os.getenv(variable)
        if not value:
            continue
        compiler = _resolve_executable(value)
        if compiler is None:
            raise RuntimeError(f"{variable} does not name an executable compiler: {value}")
        accepted, diagnostic = _probe_cuda_host_compiler(nvcc_bin, compiler)
        if not accepted:
            raise RuntimeError(
                f"NVCC rejected the compiler selected by {variable}: {compiler}: {diagnostic}"
            )
        print(f"Selected CUDA host compiler from {variable}: {compiler}")
        return compiler

    candidate_names = ["c++", "g++", *(f"g++-{major}" for major in range(30, 4, -1))]
    candidates: list[pathlib.Path] = []
    for name in candidate_names:
        compiler = _resolve_executable(name)
        if compiler is not None and compiler not in candidates:
            candidates.append(compiler)

    rejected: list[str] = []
    for compiler in candidates:
        accepted, diagnostic = _probe_cuda_host_compiler(nvcc_bin, compiler)
        if accepted:
            print(f"Selected NVCC-compatible CUDA host compiler: {compiler}")
            for entry in rejected:
                print(f"  Rejected CUDA host compiler: {entry}")
            return compiler
        rejected.append(f"{compiler}: {diagnostic}")

    detail = "\n  ".join(rejected) if rejected else "no C++ compiler was found"
    raise RuntimeError(
        "No NVCC-compatible host C++ compiler is available. Install a compiler supported "
        "by the active CUDA toolkit or set COMFY_KITCHEN_CUDA_HOST_COMPILER.\n  " + detail
    )


def _matching_c_compiler(cxx_compiler: pathlib.Path) -> pathlib.Path | None:
    name = cxx_compiler.name
    if name.startswith("g++"):
        c_name = "gcc" + name.removeprefix("g++")
    elif name.startswith("clang++"):
        c_name = "clang" + name.removeprefix("clang++")
    else:
        return None
    candidate = cxx_compiler.with_name(c_name)
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


class CMakeBuildExt(build_ext):
    # Add custom command-line options
    user_options: ClassVar = [
        *build_ext.user_options,
        ('cuda-archs=', None, 'CUDA architectures to build for (semicolon-separated, e.g., "80;89;90a")'),
        ('debug-build', None, 'Build in debug mode with debug symbols'),
        ('lineinfo', None, 'Enable NVCC line information for profiling (adds -lineinfo flag)'),
    ]

    # Default values for options
    DEFAULT_CUDA_ARCHS_WINDOWS = "75-real;75-virtual;80;89;120f"  # No need for Datacenter GPUs
    DEFAULT_CUDA_ARCHS_LINUX = "75-real;75-virtual;80;89;90a;100f;120f"  # + H100, B100

    def initialize_options(self):
        super().initialize_options()
        # Set defaults - can be overridden by command-line arguments
        self.cuda_archs = None  # Will use platform-specific default in finalize_options
        self.debug_build = False  # Default: Release build
        self.lineinfo = False  # Default: disabled

    def finalize_options(self):
        super().finalize_options()

        # Apply platform-specific default for CUDA architectures if not specified
        if self.cuda_archs is None:
            self.cuda_archs = (
                self.DEFAULT_CUDA_ARCHS_WINDOWS if os.name == "nt"
                else self.DEFAULT_CUDA_ARCHS_LINUX
            )


    def run(self):
        try:
            subprocess.run(["cmake", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("CMake must be installed to build this package") from e

        cmake_extensions = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
        regular_extensions = [ext for ext in self.extensions if not isinstance(ext, CMakeExtension)]

        for ext in cmake_extensions:
            self.build_cmake(ext)

        if regular_extensions:
            original_extensions = self.extensions
            self.extensions = regular_extensions
            super().run()
            self.extensions = original_extensions

    def build_cmake(self, ext: CMakeExtension):
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        ext_dir = ext_fullpath.parent
        ext_dir.mkdir(parents=True, exist_ok=True)

        build_temp = pathlib.Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Clean CMake cache if it exists (to avoid stale configuration)
        cmake_cache = build_temp / "CMakeCache.txt"
        if cmake_cache.exists():
            cmake_cache.unlink()
            print(f"Cleaned stale CMake cache: {cmake_cache}")

        # All options have been set in finalize_options with proper defaults
        config = "Debug" if self.debug_build else "Release"
        cuda_archs = self.cuda_archs
        enable_lineinfo = self.lineinfo

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCOMFY_CUDA_ARCHS={cuda_archs}",
            f"-DCOMFY_ENABLE_LINEINFO={'ON' if enable_lineinfo else 'OFF'}",
        ]

        cuda_home, nvcc_bin = get_cuda_path()
        cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_home}")
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_bin}")

        cuda_host_compiler = get_cuda_host_compiler(nvcc_bin)
        configure_env = os.environ.copy()
        if cuda_host_compiler is not None:
            matching_c_compiler = _matching_c_compiler(cuda_host_compiler)
            cmake_args.append(f"-DCMAKE_CXX_COMPILER={cuda_host_compiler}")
            cmake_args.append(f"-DCMAKE_CUDA_HOST_COMPILER={cuda_host_compiler}")
            configure_env["CXX"] = str(cuda_host_compiler)
            configure_env["CUDAHOSTCXX"] = str(cuda_host_compiler)
            configure_env["NVCC_CCBIN"] = str(cuda_host_compiler)
            if matching_c_compiler is not None:
                configure_env["CC"] = str(matching_c_compiler)

        build_args = ["--config", config]

        max_jobs = os.cpu_count() or 1
        # Use appropriate parallel build syntax for the platform
        if os.name == "nt":
            # Windows MSBuild uses /m:N for parallel builds
            build_args.extend(["--", f"/m:{max_jobs}"])
        else:
            # Unix make uses -jN for parallel builds
            build_args.extend(["--", f"-j{max_jobs}"])

        # Run CMake configure
        source_dir = ext.source_dir if ext.source_dir else os.path.dirname(os.path.abspath(__file__))

        print(f"Configuring CMake for {ext.name}...")
        print(f"  Source directory: {source_dir}")
        print(f"  Build directory: {build_temp}")
        print(f"  Config: {config}")
        print(f"  CUDA architectures: {cuda_archs}")
        if cuda_host_compiler is not None:
            print(f"  CUDA host compiler: {cuda_host_compiler}")
        print(f"  Line info: {'enabled' if enable_lineinfo else 'disabled'}")

        configure_cmd = ["cmake", source_dir, *cmake_args]
        try:
            subprocess.run(
                configure_cmd,
                cwd=build_temp,
                check=True,
                capture_output=False,
                env=configure_env,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake configuration failed for {ext.name}") from e

        # Run CMake build
        print(f"Building {ext.name} with CMake...")
        build_cmd = ["cmake", "--build", ".", *build_args]
        try:
            subprocess.run(
                build_cmd,
                cwd=build_temp,
                check=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake build failed for {ext.name}") from e

        print(f"Successfully built {ext.name}")

def get_cuda_path():
    nvcc_bin = None
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        nvcc_bin = pathlib.Path(cuda_home) / "bin" / "nvcc"

    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            nvcc_bin = pathlib.Path(nvcc_path)

    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_bin = pathlib.Path("/usr/local/cuda/bin/nvcc")

    if not nvcc_bin.is_file():
        return None

    if cuda_home is None:
        cuda_home = str(nvcc_bin.parent.parent)

    return cuda_home, nvcc_bin

def get_cuda_version() -> tuple[int, ...] | None:
    _cuda_home, nvcc_bin = get_cuda_path()
    try:
        output = subprocess.run(
            [nvcc_bin, "-V"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    match = re.search(r"release\s*([\d.]+)", output.stdout)
    if not match:
        return None

    version = tuple(map(int, match.group(1).split(".")))
    return version


def assert_cuda_version(version: tuple[int, ...]) -> None:
    lowest_cuda_version = (12, 8)
    if version < lowest_cuda_version:
        raise RuntimeError(
            f"ComfyKitchen CUDA backend requires CUDA {lowest_cuda_version} or newer. "
            f"Got {version}. Install will continue without CUDA backend."
        )


def setup_cuda_extension() -> CMakeExtension | None:
    print("=" * 80)
    print("Checking for CUDA availability...")
    print("=" * 80)

    if BUILD_NO_CUDA:
        print("CUDA extension disabled by --no-cuda flag")
        return None

    try:
        import nanobind  # noqa: F401
    except ImportError as e:
        raise ImportError("ERROR: nanobind not found. Install with: pip install nanobind") from e

    cuda_version = get_cuda_version()
    if cuda_version is None:
        raise RuntimeError("ERROR: Could not detect CUDA toolkit (nvcc not found). Install CUDA toolkit and try again.")

    print(f"Found CUDA version: {'.'.join(map(str, cuda_version))}")

    try:
        assert_cuda_version(cuda_version)
    except RuntimeError as e:
        raise RuntimeError(f"ERROR: {e}") from e

    root_dir = pathlib.Path(__file__).resolve().parent
    cuda_backend_dir = root_dir / "comfy_kitchen" / "backends" / "cuda"

    if not cuda_backend_dir.exists():
        raise RuntimeError(f"WARNING: CUDA backend directory not found: {cuda_backend_dir}")

    print("Building CUDA extension with CMake + nanobind: comfy_kitchen.backends.cuda._C")

    # Create CMake extension pointing to the CUDA backend directory
    ext_module = CMakeExtension(
        name="comfy_kitchen.backends.cuda._C",
        source_dir=str(cuda_backend_dir),
    )

    print("CUDA extension configured successfully (will be built with CMake)")
    return ext_module


def get_extensions() -> list[setuptools.Extension]:
    extensions = []

    if BUILD_NO_CUDA:
        print("\n" + "=" * 80)
        print("Building CPU-only variant (COMFY_KITCHEN_BUILD_NO_CUDA=1)")
        print("CUDA backend excluded - only eager, triton backends")
        print("=" * 80 + "\n")
        return extensions

    cuda_ext = setup_cuda_extension()
    if cuda_ext is not None:
        extensions.append(cuda_ext)
    else:
        print("\n" + "=" * 80)
        print("Installing comfy_kitchen without CUDA backend")
        print("Available backends: eager, triton (if installed)")
        print("=" * 80 + "\n")

    return extensions


def get_cmdclass(has_extensions):
    cmdclass = {}

    if has_extensions:
        cmdclass["build_ext"] = CMakeBuildExt

    try:
        from wheel.bdist_wheel import bdist_wheel

        class CUDABdistWheel(bdist_wheel):
            def finalize_options(self):
                super().finalize_options()
                # Set stable ABI tag only for Python 3.12+ (nanobind requirement)
                # For 3.10/3.11, leave as version-specific (cpXXX-cpXXX)
                if not BUILD_NO_CUDA and sys.version_info >= (3, 12):
                    self.py_limited_api = "cp312"

        cmdclass["bdist_wheel"] = CUDABdistWheel
    except ImportError as e:
        print(f"Warning: Could not import wheel.bdist_wheel: {e}")

    return cmdclass


def get_packages():
    if BUILD_NO_CUDA:
        cuda_dir = pathlib.Path("comfy_kitchen/backends/cuda")
        cuda_backup = pathlib.Path("cuda_backup_temp_build")

        if cuda_dir.exists():
            shutil.move(str(cuda_dir), str(cuda_backup))

        try:
            all_packages = setuptools.find_packages(where=".")
            packages = [pkg for pkg in all_packages if not pkg.startswith(("tests", "cuda_backup"))]
            return packages
        finally:
            if cuda_backup.exists():
                shutil.move(str(cuda_backup), str(cuda_dir))

    return setuptools.find_packages(where=".", exclude=["tests*"])


extensions = get_extensions()

setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": get_cmdclass(has_extensions=bool(extensions)),
}

if BUILD_NO_CUDA:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    project_meta = pyproject.get("project", {})
    version = project_meta.get("version", "0.1.0")
    description = project_meta.get("description", "")

    setup_kwargs.update({
        "packages": get_packages(),
        "name": "comfy-kitchen",
        "version": version,
        "description": f"{description} (CPU-only)",
        "include_package_data": False,
        "install_requires": [
            "torch>=2.5.0",
        ],
    })

    readme_path = pathlib.Path("README.md")
    if readme_path.exists():
        setup_kwargs.update({
            "long_description": readme_path.read_text(),
            "long_description_content_type": "text/markdown",
        })

setuptools.setup(**setup_kwargs)
