
# CodeMaxGemini: Your AI-Powered Coding Assistant
🦋README.md generated using CodeMaxGemini bot.

CodeMaxGemini is a Streamlit application that leverages Google's Gemini API to provide an AI-powered coding assistant.  This tool helps you generate code, debug, refactor, document, and perform other coding tasks efficiently.

## Features

* **Code Generation:** Generate code snippets based on your specifications.
* **Debugging:** Identify and fix errors in your code.
* **Refactoring:** Improve code structure and efficiency.
* **Documentation:** Generate code documentation.
* **Custom Requests:** Handle various coding-related tasks.
* **Gemini Model Selection:** Choose between different Gemini models (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`) for optimal performance.
* **Code Upload:** Upload code files or directly paste code into the editor.
* **GitHub Repository Integration:** Extract contents from a GitHub repository and use them to generate a README.md file. (Requires a GitHub Personal Access Token for better API rate limits)
* **Chat History:** View past interactions with the AI assistant.

## Setup

1. **Install Dependencies:**  Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

2. **Obtain a Gemini API Key:** Get your API key from [Google AI Studio](https://ai.google.com/).

3. **Run the Application:** Execute the `codeMaxGemini.py` script using Streamlit:

   ```bash
   streamlit run codeMaxGemini.py
   ```

## Usage

1. **Enter your Gemini API Key:**  The application will prompt you to enter your Gemini API key.
2. **Select a Gemini Model:** Choose the desired Gemini model from the dropdown.
3. **Choose an Action:** Select a task from the dropdown menu (e.g., "Debug Code," "Generate Code," "Generate GitHub README").
4. **Provide Input:**  Depending on the chosen action, you might need to provide code, specifications, or a GitHub repository URL.  You can upload code files or paste code directly into the integrated code editor.
5. **Send Prompt:** Click the "Send" button to submit your request to the AI assistant.
6. **View Results:** The AI assistant's response will be displayed in the chat history along with any generated or modified code.


## GitHub Repository Integration

To generate a README.md from a GitHub repository, follow these steps:

1. **Enter the Repository URL:** Provide the HTTPS URL of the GitHub repository in the designated input field.  The URL should follow the format: `https://github.com/<username>/<repository_name>.git`
2. **(Optional) Enter GitHub Personal Access Token:** This is recommended to increase the API rate limit and avoid issues when interacting with large repositories. Create a Personal Access Token in your GitHub settings (Developer settings -> Personal access tokens).  The token should have at least `repo` scope.
3. **Click "Extract Repository Contents":** This will fetch the repository's contents.
4. **Generate README:** The tool will use the extracted contents to generate a comprehensive README.md.
5. **Download README:** Download the generated `README.md` file using the provided button.

## Technologies Used

* **Streamlit:**  For the user interface.
* **streamlit-ace:** For the code editor.
* **Google Generative AI (Gemini):** For the AI capabilities.
* **Requests:** For making HTTP requests (GitHub API interaction).
