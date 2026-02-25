..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Guided Generation
=================

.. note::
   **Swift Equivalent:** This guide covers concepts that correspond to the `Generable <https://developer.apple.com/documentation/foundationmodels/generable>`_ protocol and `GenerationSchema <https://developer.apple.com/documentation/foundationmodels/generationschema>`_ in the Swift Foundation Models Framework.

Guided generation lets you constrain the model's output to follow specific structures, formats, or schemas. This ensures applications receive structured, predictable data.

Overview
--------

The Foundation Models SDK provides powerful guided generation capabilities:

* **Type-safe responses** using Python classes
* **JSON schema validation** for structured data
* **Guides** like value ranges, enums, and patterns
* **Nested structures** for complex data models

The ``@generable`` Decorator
-----------------------------

The ``@generable`` decorator transforms a Python class into a generable type equivalent to 
a Swift `Generable <https://developer.apple.com/documentation/foundationmodels/generable>`_.
Only classes decorated with ``@generable`` can be used for guided generation in 
``LanguageModelSession.respond(prompt, generating: )``. Under the hood, ``@generable`` applies 
`Python dataclasses <https://docs.python.org/3/library/dataclasses.html>`_ 
to your decorated class, so your class must be compatible with dataclass 
features for the ``@generable`` decorator to work.

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import apple_fm_sdk as fm
   from typing import List

   @fm.generable("Product review analysis")
   class ProductReview:
       sentiment: str = fm.guide("Overall sentiment", anyOf=["positive", "negative", "neutral"])
       rating: float = fm.guide("Product rating", range=(1.0, 5.0))
       keywords: List[str] = fm.guide("Key features mentioned", count=3)

   async def analyze_review():
       session = fm.LanguageModelSession(instructions="You are a product review analyzer.")

       result = await session.respond(
           "This laptop is amazing! Great performance and battery life.",
           generating=ProductReview
       )
       
       print(f"Sentiment: {result.sentiment}")  # for example, "positive"
       print(f"Rating: {result.rating}")        # for example, 4.5
       print(f"Keywords: {result.keywords}")    # for example, ["performance", "battery", "laptop"]

Available Constraints
---------------------

The ``fm.guide()`` function supports various constraints:

anyOf - Enum Values
~~~~~~~~~~~~~~~~~~~

Limit output to specific options:

.. code-block:: python
    
    import apple_fm_sdk as fm

    @fm.generable("Classification")
    class Category:
        type: str = fm.guide("Category type", anyOf=["tech", "sports", "politics"])
        priority: str = fm.guide("Priority level", anyOf=["high", "medium", "low"])

range - Numeric Ranges
~~~~~~~~~~~~~~~~~~~~~~

Constrain numbers to a specific range:

.. code-block:: python

    import apple_fm_sdk as fm

    @fm.generable("Temperature reading")
    class Temperature:
        celsius: float = fm.guide("Temperature in Celsius", range=(-50.0, 50.0))
        confidence: float = fm.guide("Confidence score", range=(0.0, 1.0))

count - List Length
~~~~~~~~~~~~~~~~~~~

Specify exact list length:

.. code-block:: python

    import apple_fm_sdk as fm
    from typing import List

    @fm.generable("Top items")
    class TopItems:
        items: List[str] = fm.guide("Top 5 items", count=5)
        tags: List[str] = fm.guide("Exactly 3 tags", count=3)

regex - Pattern Matching
~~~~~~~~~~~~~~~~~~~~~~~~~

Constrain strings to match a regex pattern. Note that the ``SystemLanguageModel`` only supports 
simple regex patterns like `\d+` for digits or `\w+` for word characters.

.. code-block:: python

    import apple_fm_sdk as fm

    @fm.generable("Contact information")
    class Contact:
        name: str = fm.guide("Full name", regex=r"\w+\s\w+")  
        age: str = fm.guide("Age", regex=r"\d+")

Complex Structures
------------------

Nested Objects
~~~~~~~~~~~~~~

Create complex nested structures:

.. code-block:: python

   import apple_fm_sdk as fm

   @fm.generable("Habitat information")
   class Habitat:
       location: str = fm.guide("Geographic location")
       climate: str = fm.guide("Climate type", anyOf=["temperate", "tropical", "arid", "polar"])
       vegetation: str = fm.guide("Primary vegetation")

   @fm.generable("Hedgehog profile")
   class Hedgehog:
       name: str = fm.guide("Hedgehog name")
       age: int = fm.guide("Age in years", range=(0, 10))
       weight: float = fm.guide("Weight in grams", range=(200.0, 1200.0))
       habitat: Habitat = fm.guide("Natural habitat")
       diet: str = fm.guide("Primary diet")

   async def extract_hedgehog_info():
       session = fm.LanguageModelSession("Extract hedgehog information from the prompt")
       
       result = await session.respond(
           "Spike is a 3-year-old hedgehog weighing 800 grams. "
           "He lives in temperate European woodlands with mixed vegetation "
           "and primarily eats insects.",
           generating=Hedgehog
       )
       
       print(f"Name: {result.name}")
       print(f"Climate: {result.habitat.climate}")
       print(f"Diet: {result.diet}")

Lists of Objects
~~~~~~~~~~~~~~~~

Generate lists of structured objects:

.. code-block:: python

    import apple_fm_sdk as fm
    from typing import List

    @fm.generable("Task item")
    class Task:
        title: str = fm.guide("Task title")
        priority: str = fm.guide("Priority", anyOf=["high", "medium", "low"])
        completed: bool = fm.guide("Completion status")

    @fm.generable("Task list")
    class TaskList:
        tasks: List[Task] = fm.guide("List of tasks", count=3)
        total_count: int = fm.guide("Total number of tasks")

    async def extract_tasks():
        session = fm.LanguageModelSession("Extract tasks from text.")

        result = await session.respond(
            "I need to: 1) Buy groceries (high priority), "
            "2) Call dentist (medium), 3) Read book (low)",
            generating=TaskList,
        )

        for task in result.tasks:
            print(f"- {task.title} [{task.priority}]")


Schema Compatibility with Swift
--------------------------------

When evaluating Swift app features, ensure your Python schemas match your Swift schemas for accurate comparison.

Swift to Python Schema Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python ``@generable`` decorator creates schemas compatible with Swift's ``Generable`` protocol:

**Swift Code:**

.. code-block:: swift

    import FoundationModels

    @Generable(description: "Product review")
    struct ProductReview {
        @Guide(description: "Sentiment")
        var sentiment: String
        
        @Guide(description: "Rating", .range(1.0...5.0))
        var rating: Double
        
        @Guide(description: "Keywords", .count(3))
        var keywords: [String]

**Equivalent Python Code:**

.. code-block:: python

   import apple_fm_sdk as fm
   from typing import List

   @fm.generable("Product review")
   class ProductReview:
       sentiment: str = fm.guide("Sentiment")
       rating: float = fm.guide("Rating", range=(1.0, 5.0))
       keywords: List[str] = fm.guide("Keywords", count=3)

Validating Schema Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure your Python schemas match your Swift schemas:

1. **Export Swift schemas to JSON:**

   .. code-block:: swift

      let schema = ProductReview.generationSchema
      let jsonData = try JSONEncoder().encode(schema)
      try jsonData.write(to: URL(fileURLWithPath: "schema.json"))

2. **Load and use in Python:**

   .. code-block:: python

      import json
      import apple_fm_sdk as fm

      # Load Swift schema
      with open("schema.json", "r") as f:
          swift_schema = json.load(f)

      # Use in a LanguageModelSession
      session = fm.LanguageModelSession(instructions="Generate a product review.")
      result = await session.respond(
          "This laptop is amazing! Great performance and battery life.",
          json_schema=swift_schema
      )
      


Error Handling
--------------

Handle errors specific to guided generation:

.. code-block:: python

    import apple_fm_sdk as fm
    from typing import Optional

    async def safe_guided_generation(prompt: str, MyGenerable: Optional[type] = None):
        try:
            if MyGenerable:
                return await fm.LanguageModelSession().respond(
                    prompt, generating=MyGenerable
                )
            else:
                return await fm.LanguageModelSession().respond(prompt)
        except fm.InvalidGenerationSchemaError as e:
            print(f"Schema error: {e}")
        except fm.UnsupportedGuideError as e:
            print(f"Unsupported constraint: {e}")
        except Exception as e:
            print(f"Generation error: {e}")
