..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Evaluation workflows
====================

.. note::
   **Swift Equivalent:** This guide covers Python-specific workflows for evaluating Swift apps that use the `Foundation Models framework <https://developer.apple.com/documentation/foundationmodels>`_. The transcript format is compatible with Swift's `Transcript <https://developer.apple.com/documentation/foundationmodels/transcript>`_ class.

The Foundation Models SDK for Python can help you evaluate foundation model features in your Swift app. 
This guide covers common evaluation workflows, including exporting data from Swift apps, processing 
transcripts, and batch processing patterns.

Exporting transcripts from Swift apps
--------------------------------------

To evaluate your Swift app's foundation model interactions, first export the session transcript. 
In your Swift code:

.. code-block:: swift

   import FoundationModels
   
   // After completing a session
   let transcript = session.transcript
   
   // Export to JSON
   if let jsonData = try? JSONEncoder().encode(transcript),
      let jsonString = String(data: jsonData, encoding: .utf8) {
       // Save to file or send to evaluation system
       try? jsonString.write(to: transcriptURL, atomically: true, encoding: .utf8)
   }

The transcript contains the complete history of the session, including all messages, tool calls, 
and model responses.

Processing transcripts in Python
---------------------------------

You can also produce transcripts directly in Python for testing using:

.. code-block:: python

   import apple_fm_sdk as fm

   session = fm.LanguageModelSession()
   await session.respond(prompt="Hello, how are you?")
   
   # Export transcript
   transcript = session.transcript
   transcript_dict = await transcript.to_dict()  # Convert to dictionary


Once you have exported transcripts, you can analyze them like any Python dictionary:

.. code-block:: python

    import json

    # Load transcript from Swift app
    with open(filepath, "r") as f:
        transcript_data = json.load(f)

    print("Transcript loaded successfully: ", transcript_data)

    # Analyze the session
    for entry in transcript_data["transcript"]["entries"]:
        role = entry["role"]
        content = entry["contents"]
        print(f"{role}: {content}")

Batch evaluation patterns
--------------------------

To evaluate multiple scenarios, use batch processing. Note that each inference call 
will be processed one at a time (not in parallel) at the macOS hardware level, so consider 
the time implications of large batches.

.. code-block:: python

    import apple_fm_sdk as fm
    import json

    # Load test cases
    test_cases = [
        {"prompt": "Test case 1", "expected": {...}},
        {"prompt": "Test case 2", "expected": {...}},
        # ... more test cases
    ]

    # Run batch evaluation
    results = []

    for i, test_case in enumerate(test_cases):
        session = fm.LanguageModelSession()

        try:
            if "schema" in test_case:
                result = await session.respond(
                    prompt=test_case["prompt"], json_schema=test_case["schema"]
                )  # structured generable response
            else:
                result = await session.respond(prompt=test_case["prompt"])  # string response

            # Compare with expected result
            success = test_case["expected"] in result
            results.append({"test_id": i, "success": success, "result": result})

        except Exception as e:
            results.append({"test_id": i, "success": False, "error": str(e)})

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

For a evaluation examples, see the git repository under `examples/ <https://github.com/apple/python-apple-fm-sdk/tree/main/examples>`_.
