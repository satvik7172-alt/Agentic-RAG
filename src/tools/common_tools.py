"""
Common tools for document and description processing.
"""

from src.llms.openai import llm


# def enhance_description_with_llm(user_description: str) -> str:
#     """
#     Enhance user-provided document description using LLM.

#     Rewrites the description to be suitable as a retriever tool instruction
#     that clearly indicates the tool is only for answering questions about
#     the uploaded content.

#     Args:
#         user_description: The original user-provided description.

#     Returns:
#         Enhanced description formatted as a tool instruction.
#     """
#     prompt = f"""
#     Rewrite the following user-provided document description to be used as a retriever tool instruction.
#     It should clearly state that the tool is only for answering questions about the uploaded content.

#     Description: "{user_description}"

#     Tool Instruction:"""

#     response = llm.invoke(prompt)
#     return response.content.strip()

# """
# Common tools for document and description processing.
# """


def enhance_description_with_llm(user_description: str) -> str:
    """
    Format the user-provided document description for use as a retriever
    tool instruction.

    Args:
        user_description: The original user-provided description.

    Returns:
        Formatted description.
    """
    description = user_description.strip()

    if not description:
        return (
            "Use this tool only to answer questions about the uploaded document."
        )

    return (
        "Use this tool only to answer questions about the uploaded document "
        f"described as: {description}"
    )