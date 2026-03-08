# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

from .generable import (
    Generable,
    GenerationSchema,
    GeneratedContent,
    GenerationID,
    ConvertibleFromGeneratedContent,
)
from .generation_property import Property
from .errors import InvalidGenerationSchemaError
from dataclasses import dataclass, field
from typing import (
    Optional,
    Union,
    get_type_hints,
    get_args,
    Type,
    List,
    Callable,
    overload,
)
import logging

logger = logging.getLogger(__name__)


class GenerableDecoratorError(InvalidGenerationSchemaError):
    """Error raised when the @fm.generable decorator is used incorrectly."""

    pass


# Overload signatures for type checkers
@overload
def generable(arg: Type[object], /) -> Type[Generable]:
    """When used without parentheses: @generable"""
    ...


@overload
def generable(arg: None = ..., /) -> Callable[[Type[object]], Type[Generable]]:
    """When used with empty parentheses: @generable()"""
    ...


@overload
def generable(arg: str, /) -> Callable[[Type[object]], Type[Generable]]:
    """When used with a description: @generable("description")"""
    ...


def generable(
    arg: Optional[Union[type, str]] = None,
) -> Union[Type[Generable], Callable[[Type], Type[Generable]]]:
    """
    Decorator that makes a class generable for use with Foundation Models.

    This decorator transforms a regular Python class (typically a dataclass) into
    a generable type that can be used with Foundation Models' guided generation
    features. It adds all necessary methods and attributes to support schema
    generation, content conversion, and partial generation during streaming.

    The decorator performs the following transformations:

    1. Converts the class to a dataclass if it isn't already
    2. Adds a ``generation_schema()`` class method for schema introspection
    3. Adds ``ConvertibleFromGeneratedContent`` support for deserialization
    4. Adds ``ConvertibleToGeneratedContent`` support for serialization
    5. Creates a ``PartiallyGenerated`` inner class for streaming support
    6. Adds required methods for structured generation

    This decorator can be used with or without parentheses:
    - ``@fm.generable`` - without parentheses
    - ``@fm.generable()`` - with empty parentheses
    - ``@fm.generable("description")`` - with a description

    :param arg: Either a class (when used without parentheses) or an optional
        human-readable description of what this type represents. This description
        is included in the generation schema and can help guide the model's
        generation behavior.
    :type arg: Optional[Union[type, str]]
    :return: Either the decorated class (when used without parentheses) or a
        decorator function (when used with parentheses)
    :rtype: Union[Type[Generable], Callable[[Type], Type[Generable]]]

    Example:
        Basic usage with a dataclass::

            import apple_fm_sdk as fm

            @fm.generable("A cat's profile")
            class Cat:
                name: str = fm.guide("Cat's name")
                age: int = fm.guide("Age in years", range=(0, 20))
                profile: str = fm.guide("What makes this cat unique")

        Usage without parentheses::

            @fm.generable
            class Dog:
                name: str
                breed: str

        Using with Session for guided generation::

            session = fm.LanguageModelSession()
            cat = await session.respond(
                Cat,
                prompt="Generate a cat named Maomao who is 2 years old and has a fluffy tail"
            )
            print(f"{cat.name} is {cat.age} years old: {cat.profile}")

        Nested generable types::

            import apple_fm_sdk as fm

            @fm.generable("Pet club")
            class PetClub:
                name: str = fm.guide("Club name")
                cats: [Cat] = fm.guide("List of cats in the club")

    .. note::
        @fm.generable automatically applies the ``@dataclass`` decorator if the
        class is not already a dataclass.

    .. seealso::
        :func:`guide` for adding constraints to individual fields.
        :class:`GenerationSchema` for the schema representation.
        :class:`Session` for using generable types in generation.
    """

    # If arg is a class, we're being used without parentheses: @generable
    if isinstance(arg, type):
        return _apply_generable_decorator(arg, description=None)

    # Otherwise, we're being used with parentheses: @generable() or @generable("description")
    description = arg

    def decorator(cls: type) -> type[Generable]:
        return _apply_generable_decorator(cls, description=description)

    return decorator


def _apply_generable_decorator(
    cls: type, description: Optional[str]
) -> type[Generable]:
    """
    Internal function that applies the generable transformation to a class.

    :param cls: The class to transform
    :param description: Optional description for the generable type
    :return: The transformed class
    """
    # Validate that we're decorating a class
    if not isinstance(cls, type):
        raise GenerableDecoratorError(
            f"@fm.generable can only be applied to classes, not {type(cls).__name__}.\n\n"
            "Correct usage:\n"
            "  @fm.generable\n"
            "  class MyClass:\n"
            "      field: str\n\n"
            "Or with a description:\n"
            "  @fm.generable('A description of MyClass')\n"
            "  class MyClass:\n"
            "      field: str"
        )

    # Validate that the class has type annotations
    if not hasattr(cls, "__annotations__") or not cls.__annotations__:
        raise GenerableDecoratorError(
            f"@fm.generable requires the class '{cls.__name__}' to have type-annotated fields.\n\n"
            "Correct usage:\n"
            "  @fm.generable\n"
            f"  class {cls.__name__}:\n"
            "      name: str  # Type annotation is required\n"
            "      age: int   # Type annotation is required\n\n"
            "Incorrect usage:\n"
            f"  class {cls.__name__}:\n"
            "      name = ''  # Missing type annotation\n"
            "      age = 0    # Missing type annotation"
        )

    # Convert to dataclass if not already
    try:
        if not hasattr(cls, "__dataclass_fields__"):
            cls = dataclass(cls)
    except Exception as e:
        raise GenerableDecoratorError(
            f"Failed to convert '{cls.__name__}' to a dataclass: {e}\n\n"
            "The @fm.generable decorator requires classes to be compatible with @dataclass.\n"
            "Common issues:\n"
            "  - Fields must have type annotations\n"
            "  - Mutable default values (like lists or dicts) must use field(default_factory=...)\n"
            "  - Class must not have conflicting __init__ or other special methods\n\n"
            "Example of correct usage:\n"
            "  from dataclasses import field\n"
            "  import apple_fm_sdk as fm\n\n"
            "  @fm.generable\n"
            f"  class {cls.__name__}:\n"
            "      name: str\n"
            "      tags: list[str] = field(default_factory=list)  # Use field() for mutable defaults"
        ) from e

    # Validate field types are supported
    try:
        get_type_hints(cls, localns={cls.__name__: cls}, include_extras=True)
    except Exception as e:
        raise GenerableDecoratorError(
            f"Failed to resolve type hints for '{cls.__name__}': {e}\n\n"
            "This usually happens when:\n"
            "  - Forward references are not properly quoted\n"
            "  - Type annotations use undefined types\n"
            "  - Circular imports prevent type resolution\n\n"
            "Example of correct usage with forward references:\n"
            "  @fm.generable\n"
            f"  class {cls.__name__}:\n"
            "      name: str\n"
            "      parent: Optional['MyClass'] = None  # Quote self-references"
        ) from e

    # Store generable metadata.
    # We need _generable as an alternative to protocols for certain dynamic type scenarios.
    cls._generable = True
    cls._generable_description = description

    cls.generation_schema = classmethod(
        generation_schema
    )  # makes schema generation a class method

    # Add ConvertibleFromGeneratedContent support
    cls._from_generated_content = classmethod(_from_generated_content)

    # Add ConvertibleToGeneratedContent support
    cls.generated_content = property(generated_content)

    # Create PartiallyGenerated inner class
    try:
        cls.PartiallyGenerated = create_partially_generated(cls)
    except Exception as e:
        raise GenerableDecoratorError(
            f"Failed to create PartiallyGenerated class for '{cls.__name__}': {e}\n\n"
            "This is an internal error. Please ensure:\n"
            "  - All field types are properly annotated\n"
            "  - Field types are serializable (str, int, float, bool, list, dict, or other @fm.generable types)\n"
            "  - No unsupported types like datetime, custom objects without @fm.generable, etc."
        ) from e

    return cls


# MARK: - Schema Helpers


def resolve_referenced_generables(
    field_type, outer_class_name: str
) -> Optional[List[GenerationSchema]]:
    """
    Resolve nested generable types referenced by a field.

    This helper function recursively examines a field's type to find any nested
    generable types (for example, a field of type ``Cat`` where ``Cat`` is itself
    a generable class). It handles collections (List, Optional) and prevents
    infinite recursion for self-referential types.

    :param field_type: The type annotation of the field to examine
    :type field_type: Type
    :param outer_class_name: Name of the outer class to detect self-references
    :type outer_class_name: str
    :return: List of GenerationSchema objects for nested generable types, or None
        if no nested generables are found or if a self-reference is detected
    :rtype: Optional[List[GenerationSchema]]

    .. note::
        This function is used internally by the schema generation process to
        build the complete schema graph including nested types.
    """
    # Check if the field_type is a generable class itself
    if hasattr(field_type, "_generable") and field_type._generable is True:
        if field_type.__name__ == outer_class_name:
            return None  # Avoid infinite recursion on self-references
        schema = field_type.generation_schema()
        return [
            schema,
            *schema.dynamic_nested_types,
        ]  # Include nested references

    # Unpack collections or optional types to find generable inner types
    for inner_type in get_args(field_type):
        return resolve_referenced_generables(inner_type, outer_class_name)


def generation_schema(cls_inner, description: Optional[str] = None) -> GenerationSchema:
    """
    Generate a GenerationSchema from a generable class.

    This function introspects a generable class to create a complete schema
    representation including all properties, their types, descriptions, guides,
    and any nested generable types. The schema can then be used for guided
    generation with Foundation Models.

    :param cls_inner: The generable class to create a schema for
    :type cls_inner: Type
    :param description: Optional description override. If not provided, uses
        the description from the generable decorator
    :type description: Optional[str]
    :return: A GenerationSchema representing the class structure
    :rtype: GenerationSchema

    .. note::
        This function is typically called automatically via the class method
        added by the generable decorator. Users don't usually need to call
        this directly.

    .. seealso::
        :func:`generable` decorator which adds this as a class method.
        :class:`GenerationSchema` for the schema representation.
    """
    properties = []
    referenced_schemas: list[GenerationSchema] = []
    referenced_schema_names: set[str] = set()
    type_hints = get_type_hints(
        cls_inner, localns={cls_inner.__name__: cls_inner}, include_extras=True
    )  # Namespace annotation needed for self-referential types

    for field_name, field_info in cls_inner.__dataclass_fields__.items():
        field_type = type_hints.get(field_name, str)

        # Get any referenced generable types
        reference = resolve_referenced_generables(field_type, cls_inner.__name__)
        if reference:
            for schema in reference:
                # Add only unique schemas to avoid duplicate types
                if schema.type_class.__name__ not in referenced_schema_names:
                    referenced_schema_names.add(schema.type_class.__name__)
                    referenced_schemas.append(schema)

        # Get description and guides from field metadata
        field_description = None
        field_guides = []
        if hasattr(field_info, "metadata") and field_info.metadata:
            field_description = field_info.metadata.get("description")
            field_guides = field_info.metadata.get("guides", [])

        prop = Property(
            name=field_name,
            type_class=field_type,
            description=field_description,
            guides=field_guides,
        )
        properties.append(prop)

    return GenerationSchema(
        type_class=cls_inner,
        description=description,
        properties=properties,
        dynamic_nested_types=referenced_schemas,
    )


# MARK: - GeneratedContent Helpers


# Add ConvertibleFromGeneratedContent support
def _from_generated_content(cls_inner, content: GeneratedContent):
    """Create instance from GeneratedContent."""
    kwargs = {}
    type_hints = get_type_hints(cls_inner)

    for field_name in cls_inner.__dataclass_fields__:
        try:
            field_type = type_hints[field_name]
            value = content.value(field_type, for_property=field_name)
            kwargs[field_name] = value
        except Exception as error:
            raise ValueError(
                f"Failed to convert GeneratedContent to {cls_inner.__name__}: "
                f"could not set field '{field_name}' with error: {error}"
            )

    return cls_inner(**kwargs)


# Add ConvertibleToGeneratedContent support
def generated_content(self) -> GeneratedContent:
    """Convert this instance to GeneratedContent."""
    content_dict = {}
    for field_name in self.__dataclass_fields__:
        content_dict[field_name] = getattr(self, field_name)
    return GeneratedContent(content_dict)


# MARK: - PartiallyGenerated Helpers


# Add _from_generated_content to PartiallyGenerated
def partial_from_generated_content(cls, partial_cls, content: GeneratedContent):
    """Create partial instance from GeneratedContent."""
    kwargs: dict = {"id": content.id}
    for field_name in cls.__dataclass_fields__:
        try:
            field_type = get_type_hints(cls)[field_name]
            value = content.value(field_type, for_property=field_name)
            kwargs[field_name] = value
        except Exception as e:
            # Field not available - leave as None
            logger.debug(f"Field '{field_name}' not available in partial content: {e}")
            kwargs[field_name] = None
    return partial_cls(**kwargs)


def create_partially_generated(cls) -> Type:
    # Create PartiallyGenerated inner class
    partial_fields = {}
    partial_annotations = {}
    type_hints = get_type_hints(
        cls, localns={cls.__name__: cls}
    )  # Namespace annotation needed for self-referential types

    for field_name, field_info in cls.__dataclass_fields__.items():
        field_type = type_hints.get(field_name, str)

        # Make all fields optional for partial generation
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            # Already optional
            partial_annotations[field_name] = field_type
        else:
            # Make optional
            partial_annotations[field_name] = Optional[field_type]

        # All fields get default None for partial generation
        partial_fields[field_name] = field(default=None)

    # Add ID field for partial generation
    partial_annotations["id"] = GenerationID
    partial_fields["id"] = field(default_factory=GenerationID)

    # Create the PartiallyGenerated class
    partial_class = type(
        f"{cls.__name__}PartiallyGenerated",
        (ConvertibleFromGeneratedContent,),
        {
            "__annotations__": partial_annotations,
            "__module__": cls.__module__,
            "_from_generated_content": classmethod(partial_from_generated_content),
            **partial_fields,
        },
    )
    partial_class = dataclass(partial_class)
    return partial_class
