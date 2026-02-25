..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Tools and function calling
==========================

.. note::
   **Swift Equivalent:** This guide covers concepts that correspond to the `Tool <https://developer.apple.com/documentation/foundationmodels/tool>`_ protocol in the Swift Foundation Models Framework.

Tools allow the model to call Python functions to access external data, perform calculations,
or interact with APIs. This enables building powerful AI agents that can take actions.

Overview
--------

The Foundation Models SDK supports tool calling through:

* **Custom tool classes** that define callable functions
* **Automatic schema generation** from function signatures
* **Type-safe parameters** using guided generation
* **Async support** for I/O operations

Creating a tool
---------------

Basic tool
~~~~~~~~~~

Create a tool by subclassing ``fm.Tool`` and implementing the ``call`` method:

.. code-block:: python

   import apple_fm_sdk as fm

   class WeatherTool(fm.Tool):
       name = "WeatherTool"
       description = "Provides weather information for a given location and units."

       @fm.generable("Weather query parameters")
       class Arguments:
           location: str = fm.guide("City name")
           units: str = fm.guide("Temperature units", anyOf=["celsius", "fahrenheit"])
       
       @property
       def arguments_schema(self) -> fm.GenerationSchema:
           return self.Arguments.generation_schema()
       
       async def call(self, args: fm.GeneratedContent) -> str:
           # Extract arguments
           location = args.value(str, for_property="location")
           units = args.value(str, for_property="units")
           
           # Simulate API call
           temp = 72 if units == "fahrenheit" else 22
           return f"The weather in {location} is {temp}°{units[0].upper()}"

Using tools with sessions
--------------------------

Register tools with a session to make them available to the model:

.. code-block:: python

    import apple_fm_sdk as fm

    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant with access to tools.",
        tools=[WeatherTool()]  # See above for WeatherTool definition
    )
    
    # The model can now call the tool
    response = await session.respond(
        "What's the weather like in Taipei?"
    )
    print(response)

Error handling
--------------

Handle tool-specific errors:

.. code-block:: python

    import apple_fm_sdk as fm

    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant with access to tools.",
        tools=[WeatherTool()] # See above for WeatherTool definition
    )

    try:
        response = await session.respond("Use the weather tool to get the current temperature in Taipei.")
    except fm.ToolCallError as e:
        print(f"Tool error: {e.tool_name} - {e.underlying_error}")
    except Exception as e:
        print(f"Error: {e}")
