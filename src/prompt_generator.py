from abc import ABC
import os
from .utils import read_file

TEMPLATE_STATEMENT = """
You write custom {} kernels to replace the pytorch operators in the given architecture to get speedups.\n
You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom {} kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

TEMPLATE_INSTRUCTION = """
Optimize the architecture named Model with custom {} kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

TEMPLATE_STATEMENT_CLEANED = """
You write custom {} kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom {} kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

TEMPLATE_INSTRUCTION_CLEANED = """
Optimize the architecture named Model with custom {} kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

PROMPT_REGISTRY = {}

def register_prompt(name: str):
    """Decorator to register a prompt generator class for a backend."""
    def decorator(cls):
        PROMPT_REGISTRY[name.lower()] = cls()
        return cls
    return decorator

class BackendPromptGenerator(ABC):
    """
    Abstract base class for generating prompts for any compute backend.
    Problem statement and instruction are generated dynamically.
    """
    def __init__(self, backend_name: str ):
        self.backend_name = backend_name

    @property
    def problem_statement(self) -> str:
        return TEMPLATE_STATEMENT.format(self.backend_name)

    @property
    def instruction(self) -> str:
        return TEMPLATE_INSTRUCTION.format(self.backend_name)

    @property
    def problem_statement_cleaned(self) -> str:
        return TEMPLATE_INSTRUCTION.format(self.backend_name)
    
    @property
    def instruction_cleaned(self) -> str:
        return TEMPLATE_INSTRUCTION.format(self.backend_name)

    def prompt_generate_custom(self, arch_src: str, example_arch_src: str = "", example_new_arch_src: str = "") -> str:
        prompt = self.problem_statement

        if example_arch_src != "" and example_new_arch_src != "":
            prompt += f"""
            Here's an example to show you the syntax of inline embedding custom {self.backend_name} kernels in torch: The example given architecture is: \n
            ``` \n
            {example_arch_src}
            ``` \n
            The example new arch with custom {self.backend_name} kernels looks like this: \n
            ```
            {example_new_arch_src}
            ``` \n
            """

        prompt += f"""
        You are given the following architecture: \n
        ```
        {arch_src}
        ```
        """
        prompt += self.instruction
        return prompt

    def prompt_generate_custom_from_prompt_template(self, ref_arch_src: str) -> str:
        """
        Using prompt example for PyKernel
        """
        arch = ref_arch_src

        example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
        example_new_arch_path = os.path.join(
            REPO_TOP_PATH, f"src/prompts/model_new_ex_add_{self.backend_name.lower()}.py"
        )

        if not os.path.exists(example_arch_path):
            raise FileNotFoundError(
                f"Example architecture file not found: {example_arch_path}"
            )
        if not os.path.exists(example_new_arch_path):
            # For now, use a basic template without examples if file doesn't exist
            return self.prompt_generate_custom(arch, "", "")

        example_arch = read_file(example_arch_path)
        example_new_arch = read_file(example_new_arch_path)

        return self.prompt_generate_custom(arch, example_arch, example_new_arch)


    def prompt_fix_compile(self, ref_arch_src, custom_kernel, metadata):
        prompt = self.problem_statement
        prompt += f"""
        With the following architecture:
        ```
        {ref_arch_src}
        ```
        You generated the following solution and it failed to compile:
        ```
        {custom_kernel}
        ```
        Here's the metadata of the compilation error:
        ```
        {metadata}
        ```
        
        Please fix the compilation error in the new model code. Please output the corrected code in codeblocks.
        """
        return prompt

    def prompt_fix_correctness(self, ref_arch_src, custom_kernel, metadata):
        prompt = self.problem_statement
        prompt += f"""
        With the following architecture:
        ```
        {ref_arch_src}
        ```
        You generated the following solution and it failed correctness:
        ```
        {custom_kernel}
        ```
        Here's the metadata of the correctness error:
        ```
        {metadata}
        ```
        Please consider how your custom {self.backend_name} kernels are implemented, how it is different from the reference implementation, and fix the correctness error in the new model code. Please output the corrected code in codeblocks.
        """
        return prompt



