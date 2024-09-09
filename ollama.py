from langchain_community.llms import Ollama
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, create_react_agent
from langchain_core.output_parsers import BaseOutputParser
from typing import List
import math

# Initialize the LLM with the specified model
llm = Ollama(model="llama3.1:8b")

# Define tools for the agent
tools = [
    Tool(
        name="Search",
        func=lambda x: "Search results for: " + x,
        description="Useful for searching information on the internet"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x, {"__builtins__": None}, math.__dict__)),  # Safer eval for basic math
        description="Useful for performing mathematical calculations"
    )
]

# Define a custom output parser to handle plain text responses from the LLM
class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str):
        # Simply return the raw text without trying to parse it
        return text

# Define the CustomPromptTemplate class that includes the required variables
class CustomPromptTemplate(StringPromptTemplate):
    template = """You are an AI assistant. You have access to the following tools:
{tools}

Available tools: {tool_names}

Question: {input}

{agent_scratchpad}"""

    input_variables: List[str] = ["tools", "tool_names", "input", "agent_scratchpad"]

    def format(self, **kwargs):
        tools = kwargs.get('tools', '')
        tool_names = kwargs.get('tool_names', '')
        input_text = kwargs.get('input', '')
        agent_scratchpad = kwargs.get('agent_scratchpad', '')
        return self.template.format(tools=tools, tool_names=tool_names, input=input_text,
                                    agent_scratchpad=agent_scratchpad)

def clean_llm_output(output):
    """ Clean the LLM output from unwanted characters or phrases """
    # Remove backticks and unnecessary formatting
    output = output.replace('`', '').strip()
    return output

# New way to use the agent (React Agent)
def create_interactive_agent():
    tools_list = '\n'.join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ', '.join([tool.name for tool in tools])
    prompt_template = CustomPromptTemplate(input_variables=["tools", "tool_names", "input", "agent_scratchpad"])

    # Use the custom output parser to bypass structured parsing
    custom_output_parser = CustomOutputParser()

    # Create the react agent using the custom parser
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template, output_parser=custom_output_parser)

    print("AI Tutor is ready. Ask anything!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting the chat.")
            break

        try:
            # Get result from agent
            result = react_agent.invoke({"input": user_input, "intermediate_steps": []})

            # Ensure the result is a string, then clean it
            if isinstance(result, str):
                cleaned_result = clean_llm_output(result)

                # Print the cleaned result
                print(f"AI Tutor: {cleaned_result}")
            else:
                # If no result was returned, handle gracefully
                print("AI Tutor: Sorry, I couldn't process that.")

        except Exception as e:
            # Catch and print the exact error to understand what's causing it
            print(f"Error: {e}")

# Start the interaction loop
create_interactive_agent()
