import os
import instructor
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentOutputSchema

app = Flask(__name__)
CORS(app)

# API Key setup
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("API key is not set. Please set the API key in the environment variable OPENAI_API_KEY.")

# Initialize memory
memory = AgentMemory()
initial_message = BaseAgentOutputSchema(
    chat_message="Hello! I am your diet assistant. Ask me anything about nutrition and healthy eating."
)
memory.add_message("assistant", initial_message)

# OpenAI client setup
client = instructor.from_openai(openai.OpenAI(api_key=API_KEY))

# Custom system prompt
system_prompt_generator = SystemPromptGenerator(
    background=["This assistant is an expert in diet, nutrition, and healthy eating habits."],
    steps=["Understand the user's input about diet and provide an accurate response."],
    output_instructions=[
        "Only answer diet-related queries. If the question is unrelated to diet, politely decline to answer.",
        "Provide clear, evidence-based dietary advice.",
        "Be friendly and professional in all interactions.",
    ],
)

# Agent setup
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=system_prompt_generator,
        memory=memory,
    )
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        print("Received Request Data:", data)

        if not data or "message" not in data:
            return jsonify({"error": "Invalid request. JSON must contain 'message'."}), 400

        user_input = data["message"]
        response = agent.run(agent.input_schema(chat_message=user_input))

        return jsonify({"response": response.chat_message})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

# Remove FlaskLambda import, no need for that in Vercel deployments
# Add WSGI handler for Vercel
if __name__ == "__main__":
    app.run(debug=True)
