# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models Generable protocol behavior
"""

import apple_fm_sdk as fm
import pytest


def test_direct_subclass_raises_error():
    """Test that directly subclassing Generable raises a TypeError."""
    print("\n=== Testing Direct Subclass of Generable ===")

    with pytest.raises(TypeError) as exc_info:

        class InvalidGenerable(fm.Generable):
            """This should raise an error."""

            name: str
            age: int

    # Verify the error message
    error_message = str(exc_info.value)
    assert "Subclassing Protocol Generable is not allowed" in error_message
    assert "Use the @fm.generable() decorator instead" in error_message
    print(f"✓ Correctly raised TypeError: {error_message}")


def test_decorator_approach_works():
    """Test that using the @fm.generable() decorator works correctly."""
    print("\n=== Testing Decorator Approach ===")

    # This should work without raising an error
    @fm.generable()
    class ValidGenerable:
        """This should work correctly."""

        name: str
        age: int

    # Verify it's recognized as a Generable
    assert isinstance(ValidGenerable, fm.Generable)
    print("✓ Decorator approach works correctly")

    # Verify it has the required methods
    assert hasattr(ValidGenerable, "generation_schema")
    assert hasattr(ValidGenerable, "_from_generated_content")
    print("✓ Has required Generable methods")


def test_alt_decorator_approach_works():
    """Test that using the @fm.generable decorator (without parentheses) works correctly."""
    print("\n=== Testing Alternative Decorator Approach ===")

    # This should work without raising an error
    @fm.generable
    class ValidGenerableAlt:
        """This should work correctly."""

        name: str
        age: int

    # Verify it's recognized as a Generable
    assert isinstance(ValidGenerableAlt, fm.Generable)
    print("✓ Alternative decorator approach works correctly")

    # Verify it has the required methods
    assert hasattr(ValidGenerableAlt, "generation_schema")
    assert hasattr(ValidGenerableAlt, "_from_generated_content")
    print("✓ Has required Generable methods")


def test_decorating_dataclass_works():
    """Test that using the @fm.generable() decorator on a dataclass works correctly."""
    print("\n=== Testing Decorator on Dataclass ===")

    from dataclasses import dataclass

    @dataclass
    @fm.generable("A description of my generable")
    class ValidGenerableDataClass:
        """This should work correctly."""

        name: str
        age: int

    # Verify it's recognized as a Generable
    assert isinstance(ValidGenerableDataClass, fm.Generable)
    print("✓ Decorator on dataclass works correctly")

    # Verify it has the required methods
    assert hasattr(ValidGenerableDataClass, "generation_schema")
    assert hasattr(ValidGenerableDataClass, "_from_generated_content")
    print("✓ Has required Generable methods")


def test_decorating_dataclass_alt_works():
    """Test that using the @fm.generable decorator (without parentheses) on a dataclass works correctly."""
    print("\n=== Testing Alternative Decorator on Dataclass ===")

    from dataclasses import dataclass

    @fm.generable
    @dataclass
    class ValidGenerableDataClassAlt:
        """This should work correctly."""

        name: str
        age: int

    # Verify it's recognized as a Generable
    assert isinstance(ValidGenerableDataClassAlt, fm.Generable)
    print("✓ Alternative decorator on dataclass works correctly")

    # Verify it has the required methods
    assert hasattr(ValidGenerableDataClassAlt, "generation_schema")
    assert hasattr(ValidGenerableDataClassAlt, "_from_generated_content")
    print("✓ Has required Generable methods")


def test_error_message_content():
    """Test that the error message contains helpful information."""
    print("\n=== Testing Error Message Content ===")

    with pytest.raises(TypeError) as exc_info:

        class TestClass(fm.Generable):
            pass

    error_message = str(exc_info.value)

    # Check for key phrases in the error message
    assert "Subclassing Protocol Generable is not allowed" in error_message
    assert "@fm.generable()" in error_message or "decorator" in error_message

    print(f"✓ Error message is helpful: '{error_message}'")


def test_error_on_missing_type_annotations():
    """Test that a helpful error is raised when class fields lack type annotations."""
    print("\n=== Testing Missing Type Annotations Error ===")

    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:

        @fm.generable
        class NoAnnotations:
            name = "default"  # Missing type annotation
            age = 0  # Missing type annotation

    error_message = str(exc_info.value)
    assert "type-annotated fields" in error_message
    assert "NoAnnotations" in error_message
    assert "Type annotation is required" in error_message
    print(f"✓ Correctly raised error for missing annotations: {error_message[:100]}...")


def test_error_on_mutable_default_values():
    """Test that a helpful error is raised when using mutable defaults incorrectly."""
    print("\n=== Testing Mutable Default Values Error ===")

    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:

        @fm.generable
        class MutableDefaults:
            name: str
            tags: list = []  # Mutable default without field()

    error_message = str(exc_info.value)
    assert "dataclass" in error_message.lower()
    assert (
        "default_factory" in error_message or "field(default_factory=" in error_message
    )
    assert "MutableDefaults" in error_message
    print(f"✓ Correctly raised error for mutable defaults: {error_message[:100]}...")


def test_error_on_invalid_type_hints():
    """Test that a helpful error is raised when type hints cannot be resolved."""
    print("\n=== Testing Invalid Type Hints Error ===")

    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:

        @fm.generable
        class InvalidTypeHints:
            name: str
            # Using an undefined type that will fail resolution
            invalid_field: "UndefinedType"  # type: ignore purposefully wrong usage  # noqa: F821

    error_message = str(exc_info.value)
    assert "type hints" in error_message.lower()
    assert "InvalidTypeHints" in error_message
    print(f"✓ Correctly raised error for invalid type hints: {error_message[:100]}...")


def test_error_on_non_class_decoration():
    """Test that decorator validation catches non-class types."""
    print("\n=== Testing Non-Class Decoration Error ===")

    # Test that the internal validation function raises an error for non-class types
    from apple_fm_sdk.generable_utils import _apply_generable_decorator

    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:
        _apply_generable_decorator(lambda: None, None)  # type: ignore[arg-type]

    error_message = str(exc_info.value)
    assert "can only be applied to classes" in error_message
    assert "Correct usage" in error_message
    print(f"✓ Correctly raised error for non-class: {error_message[:100]}...")


def test_error_messages_include_examples():
    """Test that error messages include helpful usage examples."""
    print("\n=== Testing Error Messages Include Examples ===")

    # Test missing annotations error includes examples
    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:

        @fm.generable
        class TestClass1:
            field = "value"

    error_message = str(exc_info.value)
    assert "Correct usage:" in error_message
    assert "Incorrect usage:" in error_message
    assert "class TestClass1:" in error_message
    print("✓ Missing annotations error includes examples")

    # Test mutable defaults error includes examples
    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:

        @fm.generable
        class TestClass2:
            items: list = []

    error_message = str(exc_info.value)
    assert "Example of correct usage:" in error_message
    assert "default_factory" in error_message  # Check for the key concept
    print("✓ Mutable defaults error includes examples")


def test_error_on_unsupported_field_types():
    """Test that the decorator provides helpful guidance about supported types."""
    print("\n=== Testing Unsupported Field Types Guidance ===")

    # The decorator itself doesn't validate specific types at decoration time,
    # but the error messages guide users toward supported types
    # Let's verify the error messages mention serializable types

    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:

        @fm.generable
        class TestClass:
            field = "no annotation"

    error_message = str(exc_info.value)
    # The error should guide users about proper usage
    assert "type" in error_message.lower() or "annotation" in error_message.lower()
    print(
        f"✓ Error messages guide users about type requirements: {error_message[:100]}..."
    )


def test_valid_class_with_field_defaults():
    """Test that valid classes with proper field() defaults work correctly."""
    print("\n=== Testing Valid Class with Field Defaults ===")

    from dataclasses import field

    @fm.generable
    class ValidWithDefaults:
        name: str
        tags: list[str] = field(default_factory=list)
        metadata: dict = field(default_factory=dict)

    # Verify it works
    assert hasattr(ValidWithDefaults, "generation_schema")
    schema = ValidWithDefaults.generation_schema()
    assert schema is not None
    print("✓ Valid class with field() defaults works correctly")


def test_valid_class_with_optional_fields():
    """Test that valid classes with Optional fields work correctly."""
    print("\n=== Testing Valid Class with Optional Fields ===")

    from typing import Optional

    @fm.generable
    class ValidWithOptional:
        name: str
        nickname: Optional[str] = None
        age: Optional[int] = None

    # Verify it works
    assert hasattr(ValidWithOptional, "generation_schema")
    schema = ValidWithOptional.generation_schema()
    assert schema is not None
    print("✓ Valid class with Optional fields works correctly")
