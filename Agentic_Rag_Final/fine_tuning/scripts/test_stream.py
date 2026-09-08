from langchain_core.messages import HumanMessage

from fine_tuning.local_llm import LocalQwenChatModel


print("=" * 60)
print("LOADING FINE-TUNED QWEN")
print("=" * 60)

llm = LocalQwenChatModel()

print("\nModel ready!")

question = input("\nEnter your question: ")

messages = [
    HumanMessage(content=question)
]

print("\nAnswer:\n")

for chunk in llm.stream_response(
    messages,
    max_new_tokens=200,
    temperature=0.7,
):
    print(chunk, end="", flush=True)

print("\n")
print("=" * 60)
print("STREAMING COMPLETE")
print("=" * 60)