import streamlit as st
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
import google.generativeai as genai
from datetime import datetime
from io import StringIO
import math
import requests

# Function to calculate time difference for chat history display
def TimeDiff(start_time: datetime, end_time: datetime) -> str:
    """Dynamically constructs a string describing the time difference.
    Args:
        - start_time (datetime): The starting timestamp.
        - end_time (datetime): The ending timestamp.
    Returns:
        - str: Time difference in days, hours, minutes, and seconds.
    """
    diff = end_time - start_time
    days = diff.days
    seconds = diff.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 3600) % 60
    diff_str_list = []
    if days > 0:
        diff_str_list.append(f"{days} {'day' if days == 1 else 'days'}")
    if hours > 0:
        diff_str_list.append(f"{hours} {'hour' if hours == 1 else 'hours'}")
    if minutes > 0:
        diff_str_list.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
    if seconds > 0:
        diff_str_list.append(f"{seconds} {'second' if seconds == 1 else 'seconds'}")
    if len(diff_str_list) > 1:
        return ", ".join(diff_str_list[:-1]) + " and " + diff_str_list[-1]
    elif len(diff_str_list) == 1:
        return diff_str_list[0]
    else:
        return "Just now"

def extract_github_repo_contents(repo_url, access_token=None):
    """
    Extract contents from a GitHub repository
    
    Args:
        repo_url (str): HTTPS URL of the GitHub repository
        access_token (str, optional): GitHub Personal Access Token
    
    Returns:
        dict: Repository contents with file paths and contents
    """
    # Parse repository owner and name from URL
    try:
        parts = repo_url.replace('https://github.com/', '').replace('.git', '').split('/')
        if len(parts) != 2:
            st.error("Invalid GitHub repository URL. Please use the format: https://github.com/owner/repo")
            return {}
        
        owner, repo = parts
        
        # GitHub API base URL
        base_url = f'https://api.github.com/repos/{owner}/{repo}/contents'
        
        # Headers for authentication
        headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if access_token:
            headers['Authorization'] = f'token {access_token}'
        
        # Function to recursively fetch repository contents
        def fetch_contents(path=''):
            contents = {}
            try:
                response = requests.get(f'{base_url}/{path}', headers=headers)
                response.raise_for_status()
                
                for item in response.json():
                    if item['type'] == 'file':
                        # Ignore certain file types and large files
                        if not any(item['name'].endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.zip', '.exe']):
                            # Download file content
                            file_response = requests.get(item['download_url'], headers=headers)
                            file_response.raise_for_status()
                            
                            # Limit file size to 100KB
                            if len(file_response.text) <= 100 * 1024:
                                contents[item['path']] = file_response.text
                    elif item['type'] == 'dir':
                        # Recursively fetch subdirectory contents
                        contents.update(fetch_contents(item['path']))
            
            except requests.RequestException as e:
                st.error(f"Error fetching repository contents: {e}")
            
            return contents
        
        # Fetch contents
        repo_contents = fetch_contents()
        
        # Prioritize certain files for README generation
        priority_files = [
            'README.md', 
            'requirements.txt', 
            'setup.py', 
            'pyproject.toml', 
            'main.py', 
            'app.py', 
            'src/main.py', 
            'src/app.py'
        ]
        
        # Reorder contents to prioritize important files
        prioritized_contents = {}
        for file in priority_files:
            if file in repo_contents:
                prioritized_contents[file] = repo_contents[file]
        
        # Add remaining files
        for file, content in repo_contents.items():
            if file not in prioritized_contents:
                prioritized_contents[file] = content
        
        return prioritized_contents
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return {}

# CoderBot class adapted for Gemini API
class CoderBot:
    """Coding Assistant Bot using Google's Gemini API."""
    def __init__(self, api_key: str, selected_model: str = "gemini-1.5-flash"):
        """Initialize the CoderBot with Gemini API.
        Args:
            - api_key (str): Gemini API key for authentication.
            - selected_model (str): Gemini model to use (default: 'gemini-1.5-flash').
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.selected_model = selected_model

        # Define the system instruction
        self.system_instruction = (
            "You are an AI coding assistant. Your role involves "
            "performing a wide range of tasks to help users program "
            "more efficiently. These tasks may include generating "
            "code, debugging, refactoring, documenting, and addressing "
            "other custom requests from users. Please adhere strictly "
            "to the user's requirements."
        )

        # Initialize conversation history in session state
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def chat(self, prompt: str):
        """Send user's prompt to Gemini and append response to conversation history.
        Args:
            - prompt (str): User's input prompt.
        """
        if prompt.strip():
            # Append user message to conversation history
            user_message = {"role": "user", "content": prompt, "timestamp": datetime.now()}
            st.session_state["messages"].append(user_message)

            # Prepare messages_to_send with system instruction as the first message
            messages_to_send = [{"role": "model", "parts": [{"text": self.system_instruction}]}]

            # Add user and assistant messages, mapping roles and structuring parts
            for msg in st.session_state["messages"]:
                role = "user" if msg["role"] == "user" else "model"  # Map "assistant" to "model"
                parts = [{"text": msg["content"]}]
                messages_to_send.append({"role": role, "parts": parts})

            # Call Gemini API
            try:
                model = genai.GenerativeModel(f"models/{self.selected_model}")
                response = model.generate_content(messages_to_send)
                if response.candidates and response.candidates[0].content.parts:
                    bot_message = response.candidates[0].content.parts[0].text
                else:
                    bot_message = "No response generated."
            except Exception as e:
                bot_message = f"Error: Could not generate response. ({str(e)})"

            # Append assistant message to conversation history
            assistant_message = {"role": "assistant", "content": bot_message, "timestamp": datetime.now()}
            st.session_state["messages"].append(assistant_message)

# App class for the Streamlit interface
class App:
    """Streamlit application for the coding assistant."""
    EXTENSIONS = {".html": "html", ".css": "css", ".js": "javascript",
                  ".py": "python", ".java": "java", ".c": "c_cpp",
                  ".cs": "csharp", ".PHP": "php", ".swift": "swift",
                  ".bas": "vba", ".txt": "plain_text"}

    def __init__(self):
        """Initialize the App."""
        st.set_page_config(
            page_title="CodeMaxGemini",
            page_icon=":computer:",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        self.bot = None
        self.LANGUAGES = LANGUAGES
        self.LANGUAGES.append("vba")
        self.LANGUAGES = list(set(self.LANGUAGES))
        if "files" not in st.session_state:
            st.session_state["files"] = {}
        if "code_language" not in st.session_state:
            st.session_state["code_language"] = ""
        if "code_theme" not in st.session_state:
            st.session_state["code_theme"] = ""
        if "code_font_size" not in st.session_state:
            st.session_state["code_font_size"] = ""

    def send_prompt(self, prompt: str):
        """Send prompt to the bot and display uploaded file names.
        Args:
            - prompt (str): User's input prompt.
        """
        if st.session_state["files"]:
            st.text("")
            for file in list(st.session_state["files"].keys())[::-1]:
                if file != "Sample Code Provided":
                    st.text(f"[{file} uploaded]")
        self.bot.chat(prompt=prompt)

    def get_code(self, initial_code: str, initial_lang: str) -> str:
        """Retrieve code from the editor.
        Args:
            - initial_code (str): Initial code for the editor.
            - initial_lang (str): Initial programming language.
        Returns:
            - str: Code entered by the user.
        """
        st.session_state["code_language"] = self.c1.selectbox(
            "Language Mode",
            options=self.LANGUAGES,
            index=self.LANGUAGES.index(initial_lang),
        )
        st.session_state["code_theme"] = self.c1.selectbox(
            "Editor Theme",
            options=THEMES,
            index=THEMES.index("tomorrow_night"),
        )
        st.session_state["code_font_size"] = self.c1.slider(
            "Font size", 5, 24, 14
        )
        with self.c2:
            code = st_ace(
                placeholder="Input your code here",
                language=st.session_state["code_language"],
                theme=st.session_state["code_theme"],
                keybinding="vscode",
                font_size=st.session_state["code_font_size"],
                tab_size=4,
                show_gutter=True,
                show_print_margin=False,
                wrap=False,
                auto_update=True,
                readonly=False,
                min_lines=45,
                key="ace",
                height=self.code_editor_height,
                value=initial_code,
            )
        return code

    def get_code_language(self, file_name: str, default_lang: str) -> str:
        """Determine code language from file extension.
        Args:
            - file_name (str): Name of the file.
            - default_lang (str): Default language if no extension is found.
        Returns:
            - str: Determined code language.
        """
        code_language = default_lang
        if file_name.strip() and ("." in file_name):
            file_extension = "." + file_name.split(".")[1]
            code_language = self.EXTENSIONS.get(file_extension, "plain_text")
        return code_language

    def send_no_code(self, user_message: str):
        """Send text prompt without code.
        Args:
            - user_message (str): User's text prompt.
        """
        if user_message.strip() and self.col1.button("Send"):
            self.send_prompt(user_message)

    def send_code(self, user_message: str, file_name: str, initial_code: str):
        """Send text prompt with code.
        Args:
            - user_message (str): User's text prompt.
            - file_name (str): Name of the file or empty string.
            - initial_code (str): Initial code for the editor.
        """
        initial_code_language = self.get_code_language(file_name=file_name, default_lang="python")
        code = self.get_code(initial_code=initial_code, initial_lang=initial_code_language)
        if code.strip():
            prompt_code = f"Here is the `{file_name}` code:  \n```{st.session_state['code_language']}  \n{code}  \n```" if file_name.strip() else f"Here is the code:  \n```{st.session_state['code_language']}  \n{code}  \n```"
            prompt = user_message + "  \n" + prompt_code
            self.c1.markdown("#")
            self.c1.markdown("#")
            self.c1.markdown("##")
            if self.c1.button("Send"):
                if file_name.strip():
                    if "Sample Code Provided" in st.session_state["files"]:
                        del st.session_state["files"]["Sample Code Provided"]
                    st.session_state["files"][file_name] = code
                else:
                    st.session_state["files"]["Sample Code Provided"] = code
                self.send_prompt(prompt)

    def upload_code(self, user_message):
        """Handle code upload in three ways.
        Args:
            - user_message (str): User's text prompt.
        """
        upload_method = self.col1.selectbox(
            label="How would you prefer to upload your code?",
            options=["Not now", "Enter / paste my code", "Upload my code script"],
        )
        if upload_method == "Not now":
            self.send_no_code(user_message)
        elif upload_method == "Enter / paste my code":
            self.c2, self.c1 = st.columns([9, 2])
            self.code_editor_height = 496
            file_name = self.c1.text_input(
                "File Name",
                placeholder="eg. index.html",
                help="Provide the filename or leave blank if not applicable.",
            )
            self.send_code(user_message=user_message, file_name=file_name, initial_code="")
        elif upload_method == "Upload my code script":
            uploaded_file = st.file_uploader("Upload a file from your local computer")
            if uploaded_file:
                file_name = uploaded_file.name
                script_code = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                self.code_editor_height = 410
                self.c2, self.c1 = st.columns([9, 2])
                self.send_code(user_message=user_message, file_name=file_name, initial_code=script_code)

    def show_code_uploaded(self):
        """Display uploaded code in expanders."""
        if st.session_state["files"]:
            self.col3.markdown("Expand to view the uploaded code below:")
            for file, code in list(st.session_state["files"].items())[::-1]:
                with self.col3.expander(label=file, expanded=False):
                    uploaded_code_language = st.session_state["code_language"] if file == "Sample Code Provided" else self.get_code_language(file_name=file, default_lang="plain_text")
                    st_ace(
                        value=code,
                        language=uploaded_code_language,
                        theme=st.session_state["code_theme"],
                        keybinding="vscode",
                        font_size=st.session_state["code_font_size"],
                        tab_size=4,
                        show_gutter=True,
                        show_print_margin=False,
                        wrap=False,
                        auto_update=True,
                        readonly=True,
                        min_lines=45,
                        key=f"ace-{file}",
                        height=300,
                    )

    def output_chat_history(self):
        """Display chat history between user and bot."""
        if "messages" in st.session_state:
            chat_messages = st.session_state["messages"]
            current_time = datetime.now()
            for msg in chat_messages[::-1]:  # Most recent first
                role = "You" if msg["role"] == "user" else "CoderBot"
                time_diff = TimeDiff(start_time=msg["timestamp"], end_time=current_time)
                time_label = f"<{time_diff} ago>"
                content = msg["content"]
                st.markdown(
                    f"<span style='color:#6699FF'><strong>{role} </strong>{time_label}:</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(content)

    def run(self):
        """Run the application."""
        st.title("Welcome to CodeMaxGemini")
        st.subheader(
            "Simplifying Coding With CodeMaxGemini: Your AI Coding Assistant For "
            "Easy Code Generation, Debugging, Refactoring, and Documentation "
            ":computer:"
        )

        cl1, cl2, cl3 = st.columns([1, 0.8, 1])
        # Gemini model selection
        MODEL = cl1.selectbox(
            "Select a Gemini model",
            ("gemini-1.5-flash", "gemini-1.5-pro"),
            help="Select the Gemini model for the coding assistant. Visit Google AI Studio for more info."
        )
        KEY = cl3.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="To create a Gemini API key, visit Google AI Studio."
        )

        st.markdown("***")

        if not KEY.strip():
            st.error("Please enter your Gemini API key to start the service!")
        else:
            self.bot = CoderBot(api_key=KEY, selected_model=MODEL)
            st.text("")
            self.col1, col2, self.col3 = st.columns([1, 0.25, 1])

            action = self.col1.selectbox(
                label="How can the bot assist with your code?",
                options=[
                    "Specify Custom Requirements",
                    "Debug Code",
                    "Refactor Code",
                    "Refactor Code to OOP",
                    "Comment Code",
                    "Review Code",
                    "Generate GitHub README",
                    "Suggest a Solution For a Coding Challenge",
                    "[Delete all previously uploaded files]",
                ],
                index=0,
            )

            if action == "Generate GitHub README":
                self.show_code_uploaded()
                
                # Add GitHub Personal Access Token input (optional)
                github_token = self.col1.text_input(
                    "GitHub Personal Access Token (optional)",
                    type="password",
                    help="Provides higher API rate limits. Create at GitHub Settings > Developer Settings > Personal Access Tokens"
                )
                
                # GitHub Repository URL input
                repo_url = self.col1.text_input(
                    "Enter HTTPS URL of a remote GitHub repo",
                    placeholder="https://github.com/<user>/<repo>.git",
                    help="Enter the full GitHub repository URL"
                )
                
                # Button to extract repository contents
                if self.col1.button("Extract Repository Contents"):
                    try:
                        # Extract repository contents
                        repo_contents = extract_github_repo_contents(repo_url, github_token)
                        
                        # Display extracted contents
                        if repo_contents:
                            st.subheader("Repository Contents")
                            content_text = "\n\n---\n\n".join([f"### {path}\n```\n{content}\n```" for path, content in repo_contents.items()])
                            
                            # Send contents to Gemini for README generation
                            prompt = (f"Generate a comprehensive GitHub README.md for this repository. "
                                      f"Use the following repository contents to understand the project:\n\n{content_text}")
                            
                            self.send_prompt(prompt)
                            
                            # Get the generated README
                            readme = [msg["content"] for msg in st.session_state["messages"] if msg["role"] == "assistant"][-1]
                            
                            # Display and download README
                            self.col1.download_button(
                                label="Download README.md",
                                data=readme,
                                file_name="README.md",
                                mime="text/markdown",
                            )
                            
                            st.markdown("Generated README:")
                            st.code(readme, language="markdown")
                        else:
                            st.error("Could not extract repository contents.")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

            # Rest of the code remains the same as in the original implementation
            elif action == "Suggest a Solution For a Coding Challenge":
                coding_langs = ["Python", "Java", "MySQL", "MS SQL Server", "Oracle SQL", "JavaScript", "C#", "C", "C++",
                                "Ruby", "Swift", "Go", "Scala", "Kotlin", "Rust", "PHP", "TypeScript", "Racket",
                                "Erlang", "Elixir", "Dart"]
                _c1, _c2 = self.col1.columns([3, 1])
                coding_problem = _c1.text_area(
                    "Input the coding challenge",
                    placeholder="Describe the challenge or paste a URL.",
                    value="",
                    height=170,
                )
                lang_selected = _c2.selectbox("Language Mode", options=coding_langs, index=0)
                prompt = (f"Solve the problem in {lang_selected}:  \n{coding_problem}  \nExplain the solution and "
                          "display it in a code block." if "SQL" in lang_selected else
                          f"Solve the problem in {lang_selected}:  \n{coding_problem}  \nExplain the solution and "
                          "display it in a code block.  \nAlso, clarify the time and space complexity.")
                if coding_problem.strip():
                    _c2.markdown("##")
                    _c2.markdown("###")
                    _c2.markdown("###")
                    if _c2.button("Send"):
                        self.send_prompt(prompt)

            else:
                if action == "Specify Custom Requirements":
                    user_message = self.col1.text_area(
                        "Specify your requirements here",
                        placeholder="Try: Build an app in Python... Write documentation...",
                        value="",
                        height=180,
                    )
                elif action == "Debug Code":
                    user_message = "Debug the code. Clarify where went wrong and what caused the error. Rewrite the code in a code block."
                elif action == "Refactor Code":
                    user_message = "Refactor the code in a more efficient way. Rewrite the code in a code block."
                elif action == "Refactor Code to OOP":
                    user_message = "Refactor the code in a more efficient way. Rewrite the code to OOP in a code block."
                elif action == "Comment Code":
                    user_message = "Add comments to the code line by line. Display all the comments and code inside a code block."
                elif action == "Review Code":
                    user_message = "Review the code. Provide feedback on issues by line number and suggest improvements. Display the updated code in a code block."
                elif action == "[Delete all previously uploaded files]":
                    st.session_state["files"] = {}
                    user_message = self.col1.text_area(
                        "Specify your requirements here",
                        value="Please disregard any previously provided code.",
                        height=180,
                    )

                self.upload_code(user_message)
                self.show_code_uploaded()

            st.text("")
            self.output_chat_history()

if __name__ == "__main__":
    app = App()
    app.run()