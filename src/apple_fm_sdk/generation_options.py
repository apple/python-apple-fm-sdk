# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class SamplingModeType(str, Enum):
    """Enumeration of available sampling mode types.

    :cvar GREEDY: Always select the most likely token
    :cvar RANDOM: Randomly select from high-probability tokens
    """

    GREEDY = "greedy"
    RANDOM = "random"


@dataclass
class SamplingMode:
    """A type that defines how values are sampled from a probability distribution.

    This class represents different sampling strategies that control how the model
    picks tokens when generating a response. The model builds its response in a loop,
    and at each iteration it produces a probability distribution for all tokens in
    its vocabulary. The sampling mode determines how to select the next token from
    this distribution.

    :ivar mode_type: The type of sampling mode
    :vartype mode_type: SamplingModeType
    :ivar top: For random sampling with fixed top-k, the number of high-probability
        tokens to consider
    :vartype top: Optional[int]
    :ivar probability_threshold: For random sampling with variable threshold, the
        cumulative probability threshold
    :vartype probability_threshold: Optional[float]
    :ivar seed: Random seed for reproducible random sampling
    :vartype seed: Optional[int]
    """

    mode_type: SamplingModeType
    top: Optional[int] = None
    probability_threshold: Optional[float] = None
    seed: Optional[int] = None

    @classmethod
    def greedy(cls) -> "SamplingMode":
        """Create a sampling mode that always chooses the most likely token.

        Greedy sampling provides deterministic, focused responses by always
        selecting the token with the highest probability at each step.

        :return: A SamplingMode configured for greedy sampling
        :rtype: SamplingMode

        Example::

            import apple_fm_sdk as fm

            sampling = fm.SamplingMode.greedy()
            options = fm.GenerationOptions(sampling=sampling)
        """
        return cls(mode_type=SamplingModeType.GREEDY)

    @classmethod
    def random(
        cls,
        top: Optional[int] = None,
        probability_threshold: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "SamplingMode":
        """Create a random sampling mode with optional constraints.

        Random sampling introduces variability in responses by randomly selecting
        from high-probability tokens. You can constrain the selection using either:

        - **top**: Consider only the top-k most likely tokens (fixed number)
        - **probability_threshold**: Consider tokens until cumulative probability
          reaches the threshold (variable number)

        :param top: Number of high-probability tokens to consider. If specified,
            only the top-k most likely tokens are candidates for selection.
        :type top: Optional[int]
        :param probability_threshold: Cumulative probability threshold (0.0 to 1.0).
            If specified, tokens are considered until their cumulative probability
            reaches this threshold.
        :type probability_threshold: Optional[float]
        :param seed: Random seed for reproducible sampling. Using the same seed
            with the same inputs will produce the same outputs.
        :type seed: Optional[int]
        :return: A SamplingMode configured for random sampling
        :rtype: SamplingMode
        :raises ValueError: If both top and probability_threshold are specified,
            or if values are out of valid ranges

        Examples:
            Random sampling with top-k::

                import apple_fm_sdk as fm

                # Consider only top 50 most likely tokens
                sampling = fm.SamplingMode.random(top=50, seed=42)
                options = fm.GenerationOptions(sampling=sampling)

            Random sampling with probability threshold::

                import apple_fm_sdk as fm

                # Consider tokens until 90% cumulative probability
                sampling = fm.SamplingMode.random(
                    probability_threshold=0.9,
                    seed=42
                )
                options = fm.GenerationOptions(sampling=sampling)

            Random sampling with seed only::

                import apple_fm_sdk as fm

                # Reproducible random sampling without constraints
                sampling = fm.SamplingMode.random(seed=42)
                options = fm.GenerationOptions(sampling=sampling)

        Note:
            - Only one of ``top`` or ``probability_threshold`` can be specified
            - If neither is specified, all tokens are considered
            - The ``seed`` parameter enables reproducible generation
        """
        if top is not None and probability_threshold is not None:
            raise ValueError(
                "Cannot specify both 'top' and 'probability_threshold'. "
                "Choose one sampling constraint."
            )

        if top is not None and (not isinstance(top, int) or top <= 0):
            raise ValueError("'top' must be a positive integer")

        if probability_threshold is not None and (
            not isinstance(probability_threshold, (int, float))
            or not 0.0 <= probability_threshold <= 1.0
        ):
            raise ValueError("'probability_threshold' must be between 0.0 and 1.0")

        if seed is not None and not isinstance(seed, int):
            raise ValueError("'seed' must be an integer")

        return cls(
            mode_type=SamplingModeType.RANDOM,
            top=top,
            probability_threshold=probability_threshold,
            seed=seed,
        )


@dataclass
class GenerationOptions:
    """Options that control how the model generates its response to a prompt.

    Generation options determine the decoding strategy the framework uses to adjust
    the way the model chooses output tokens. When you interact with the model, it
    converts your input to a token sequence and uses it to generate the response.

    **Important Considerations:**

    - Only use ``maximum_response_tokens`` when you need to protect against
      unexpectedly verbose responses. Enforcing a strict token response limit can
      lead to the model producing malformed results or grammatically incorrect
      responses.

    - All input to the model contributes tokens to the context window, including
      the Instructions, Prompt, Tool definitions, and Generable types, as well as
      the model's responses. If your session exceeds the available context size,
      it throws an ExceededContextWindowSizeError.

    :ivar sampling: A sampling strategy for how the model picks tokens when
        generating a response. Defaults to None (uses model default).
    :vartype sampling: Optional[SamplingMode]
    :ivar temperature: Temperature influences the confidence of the model's response.
        Higher values (e.g., 1.0) make output more random and creative, while lower
        values (e.g., 0.1) make it more focused and deterministic. Valid range is
        typically 0.0 to 1.0. Defaults to None (uses model default).
    :vartype temperature: Optional[float]
    :ivar maximum_response_tokens: The maximum number of tokens the model is allowed
        to produce in its response. Use this to prevent unexpectedly verbose responses,
        but be aware that strict limits may result in incomplete or malformed output.
        Defaults to None (no explicit limit).
    :vartype maximum_response_tokens: Optional[int]

    Examples:
        Default options::

            import apple_fm_sdk as fm

            options = fm.GenerationOptions()

        Custom temperature and token limit::

            import apple_fm_sdk as fm

            options = fm.GenerationOptions(
                temperature=0.7,
                maximum_response_tokens=500
            )

        Greedy sampling with temperature::

            import apple_fm_sdk as fm

            options = fm.GenerationOptions(
                sampling=fm.SamplingMode.greedy(),
                temperature=0.3
            )

        Random sampling with constraints::

            import apple_fm_sdk as fm

            options = fm.GenerationOptions(
                sampling=fm.SamplingMode.random(top=50, seed=42),
                temperature=0.8,
                maximum_response_tokens=1000
            )

    See Also:
        - :class:`SamplingMode`: For configuring sampling strategies
        - :class:`~apple_fm_sdk.session.LanguageModelSession`: For using options in sessions
    """

    sampling: Optional[SamplingMode] = None
    temperature: Optional[float] = None
    maximum_response_tokens: Optional[int] = None

    def __post_init__(self):
        """Validate generation options after initialization.

        :raises ValueError: If any option values are invalid
        """
        if self.temperature is not None:
            if not isinstance(self.temperature, (int, float)):
                raise ValueError("'temperature' must be a number")
            if self.temperature < 0.0:
                raise ValueError("'temperature' must be non-negative")

        if self.maximum_response_tokens is not None:
            if not isinstance(self.maximum_response_tokens, int):
                raise ValueError("'maximum_response_tokens' must be an integer")
            if self.maximum_response_tokens <= 0:
                raise ValueError("'maximum_response_tokens' must be positive")

        if self.sampling is not None and not isinstance(self.sampling, SamplingMode):
            raise ValueError("'sampling' must be a SamplingMode instance")

    def to_dict(self) -> dict:
        """Convert GenerationOptions to a dictionary for JSON serialization.

        This method converts the GenerationOptions instance into a dictionary
        format suitable for passing to the C bindings layer as JSON.

        :return: Dictionary representation of the generation options
        :rtype: dict

        Example::

            import apple_fm_sdk as fm

            options = fm.GenerationOptions(
                temperature=0.7,
                sampling=fm.SamplingMode.random(top=50),
                maximum_response_tokens=500
            )
            options_dict = options.to_dict()
            # {'temperature': 0.7, 'sampling': {'mode': 'random', 'top_k': 50}, 'maximum_response_tokens': 500}
        """
        result = {}

        if self.sampling is not None:
            sampling_dict = {"mode": self.sampling.mode_type.value}
            if self.sampling.mode_type == SamplingModeType.RANDOM:
                if self.sampling.top is not None:
                    sampling_dict["top_k"] = str(self.sampling.top)
                if self.sampling.probability_threshold is not None:
                    sampling_dict["top_p"] = str(self.sampling.probability_threshold)
                if self.sampling.seed is not None:
                    sampling_dict["seed"] = str(self.sampling.seed)
            result["sampling"] = sampling_dict

        if self.temperature is not None:
            result["temperature"] = self.temperature

        if self.maximum_response_tokens is not None:
            result["maximum_response_tokens"] = self.maximum_response_tokens

        return result
