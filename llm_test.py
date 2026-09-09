def get_llm_reply(message):
    # Real LLM will be connected here later
    return "LLM response will go here"

print("Haku LLM Test")
print("Type exit to stop")

while True:
    message = input("\nYou: ")

    if message.lower() == "exit":
        print("Program stopped")
        break

    if message.strip() == "":
        print("Please enter a message")
        continue

    reply = get_llm_reply(message)

    print("Haku:", reply)