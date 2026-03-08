# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import ast
import subprocess
import shutil
import platform
from setuptools import build_meta as setuptools_backend
from pathlib import Path
import re
from typing import Optional

DEFAULT_SWIFT_BUILD_CONFIGURATION = "release"


class SwiftToolingError(Exception):
    pass


def _fix_library_search_dirs(paths_str: str) -> str:
    """
    Post-process the add_library_search_dirs call to use runtime-relative paths.
    Replaces 'lib' with os.path.join(os.path.dirname(__file__), 'lib')
    """
    # Parse the list of paths (they're quoted strings separated by commas)
    try:
        paths = ast.literal_eval(f"[{paths_str}]")
    except:
        # If parsing fails, return as-is
        return f"add_library_search_dirs([{paths_str}])"

    # Replace 'lib' with runtime-relative path
    new_paths = []
    for path in paths:
        if path == "lib":
            new_paths.append("os.path.join(os.path.dirname(__file__), 'lib')")
        else:
            new_paths.append(repr(path))

    return f"add_library_search_dirs([{', '.join(new_paths)}])"


def _build_c_bindings(
    swift_build_config: str,
    override_library_search_path: Optional[str],
    override_library_name: Optional[str],
):
    print("Building C bindings...")

    # Check macOS version
    if platform.system() == "Darwin":
        macos_version = platform.mac_ver()[0]
        try:
            major, minor = map(int, macos_version.split(".")[:2])
            if major < 26:
                raise SwiftToolingError(
                    f"macOS version {macos_version} found, but version 26.0 or higher is required. "
                    "This package requires macOS 26.0+ to build the Swift bindings."
                )
        except (ValueError, IndexError) as e:
            raise SwiftToolingError(
                f"Could not parse macOS version '{macos_version}': {e}"
            )

    # Check if Swift is installed
    if shutil.which("swift") is None:
        raise SwiftToolingError(
            "No `swift` executable found in PATH. Is `swift` set up on your system?"
        )

    # Check if xcode-select points to a full Xcode installation (not just Command Line Tools)
    try:
        developer_dir = subprocess.run(
            ["xcode-select", "-p"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Command Line Tools path is typically /Library/Developer/CommandLineTools
        # Full Xcode path is typically /Applications/Xcode.app/Contents/Developer
        if "CommandLineTools" in developer_dir:
            raise SwiftToolingError(
                f"The active developer directory is set to Command Line Tools ({developer_dir}), "
                "but a full Xcode installation is required. Please install Xcode. Then open Xcode"
                "at least once to accept the license agreement and install the Swift SDKs."
            )
    except subprocess.CalledProcessError as e:
        raise SwiftToolingError(
            f"Failed to check active developer directory: {e.stderr if e.stderr else str(e)}"
        )

    # Check if Xcode is installed and version >= 26.0
    if shutil.which("xcodebuild") is None:
        raise SwiftToolingError(
            "No `xcodebuild` executable found in PATH. Is Xcode installed on your system?"
        )

    try:
        xcode_version_output = subprocess.run(
            ["xcodebuild", "-version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Parse version from output like "Xcode 26.0" or "Xcode 26.0.1"
        version_match = re.search(r"Xcode\s+(\d+)\.(\d+)", xcode_version_output)
        if not version_match:
            raise SwiftToolingError(
                f"Could not parse Xcode version from: {xcode_version_output}"
            )

        major_version = int(version_match.group(1))
        minor_version = int(version_match.group(2))

        if major_version < 26:
            raise SwiftToolingError(
                f"Xcode version {major_version}.{minor_version} found, but version 26.0 or higher is required."
            )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        # Provide helpful message if the error is about Command Line Tools
        if "command line tools instance" in error_msg.lower():
            raise SwiftToolingError(
                "Xcode is not properly configured. The active developer directory points to "
                "Command Line Tools instead of a full Xcode installation. Please install Xcode "
                "from the App Store and run 'sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer'"
            )
        raise SwiftToolingError(f"Failed to check Xcode version: {error_msg}")

    swiftPackageDir = Path("foundation-models-c").resolve()

    try:
        result = subprocess.run(
            ["swift", "build", "-c", swift_build_config],
            check=True,
            cwd=str(swiftPackageDir),
            capture_output=True,
            text=True,
        )
        build_dir_string = subprocess.run(
            ["swift", "build", "-c", swift_build_config, "--show-bin-path"],
            check=True,
            cwd=str(swiftPackageDir),
            capture_output=True,
            text=True,
        ).stdout.strip()
        libraryDirectory = (Path("src") / "apple_fm_sdk" / "lib").resolve()

        build_dir = Path(build_dir_string)
        if libraryDirectory.exists():
            shutil.rmtree(str(libraryDirectory))
        shutil.copytree(str(build_dir), str(libraryDirectory))

        ctypesgen_library_lookup_args = [
            "-L",
            "lib",  # Relative to the apple_fm_sdk package directory
            "-l",
            "FoundationModels",
        ]
        if override_library_search_path:
            ctypesgen_library_lookup_args.append("-L")
            ctypesgen_library_lookup_args.append(override_library_search_path)
        if override_library_name:
            ctypesgen_library_lookup_args.append("-l")
            ctypesgen_library_lookup_args.append(override_library_name)

        subprocess.run(
            [
                "ctypesgen",
                str(
                    swiftPackageDir
                    / "Sources"
                    / "FoundationModelsCBindings"
                    / "include"
                    / "FoundationModels.h"
                ),
                *ctypesgen_library_lookup_args,
                "-o",
                "./src/apple_fm_sdk/_ctypes_bindings.py",
            ],
            check=True,
        )

        # Post-process the generated file to use runtime-relative paths
        bindings_file = Path("src/apple_fm_sdk/_ctypes_bindings.py")
        bindings_content = bindings_file.read_text()

        # Ensure 'os' module is imported
        if "import os" not in bindings_content:
            # Add import after the existing imports (after ctypes and sys imports)
            bindings_content = bindings_content.replace(
                "import sys\n", "import sys\nimport os\n"
            )

        # Replace the add_library_search_dirs call to use path relative to the package
        # Find patterns like: add_library_search_dirs(['lib', '/path/to/other'])
        # Replace 'lib' with os.path.join(os.path.dirname(__file__), 'lib')
        bindings_content = re.sub(
            r"add_library_search_dirs\(\[(.*?)\]\)",
            lambda m: _fix_library_search_dirs(m.group(1)),
            bindings_content,
        )

        bindings_file.write_text(bindings_content)
    except subprocess.CalledProcessError as e:
        # Capture both stdout and stderr for comprehensive error reporting
        error_parts = []
        if e.stdout:
            error_parts.append(f"stdout:\n{e.stdout}")
        if e.stderr:
            error_parts.append(f"stderr:\n{e.stderr}")

        if error_parts:
            error_msg = "\n".join(error_parts)
        else:
            error_msg = f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}"

        raise SwiftToolingError(
            f"Failed to build Swift-C-Python bindings:\n{error_msg}"
        )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    swift_build_config = DEFAULT_SWIFT_BUILD_CONFIGURATION
    override_library_search_path = None
    override_library_name = None
    if config_settings:
        if "swift-build-config" in config_settings:
            swift_build_config = config_settings["swift-build-config"]
        if "override-library-name" in config_settings:
            override_library_name = config_settings["override-library-name"]
        if "override-library-search-path" in config_settings:
            override_library_search_path = config_settings[
                "override-library-search-path"
            ]

    _build_c_bindings(
        swift_build_config, override_library_search_path, override_library_name
    )
    return setuptools_backend.build_wheel(
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    swift_build_config = DEFAULT_SWIFT_BUILD_CONFIGURATION
    override_library_search_path = None
    override_library_name = None
    if config_settings:
        if "swift-build-config" in config_settings:
            swift_build_config = config_settings["swift-build-config"]
        if "override-library-name" in config_settings:
            override_library_name = config_settings["override-library-name"]
        if "override-library-search-path" in config_settings:
            override_library_search_path = config_settings[
                "override-library-search-path"
            ]

    _build_c_bindings(
        swift_build_config, override_library_search_path, override_library_name
    )
    return setuptools_backend.build_editable(
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def get_requires_for_build_editable(config_settings=None):
    return ["setuptools>=64", "ctypesgen"]


def get_requires_for_build_wheel(config_settings=None):
    return ["setuptools>=64", "wheel", "ctypesgen"]


def build_sdist(sdist_directory, config_settings=None):
    """
    Build source distribution without compiling Swift/C code.
    This function intentionally does NOT call _build_c_bindings().
    """
    # Explicitly bypass any compilation - just package source files
    return setuptools_backend.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_sdist(config_settings=None):
    """
    Requirements for building sdist - no Swift tooling or ctypesgen needed.
    """
    return ["setuptools>=64"]


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return setuptools_backend.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )
