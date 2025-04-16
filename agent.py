import anthropic
import os
import subprocess
import re
import logging
import tempfile
import sys
from datetime import datetime
import time
import select

# --- Configuration ---

# Parent directory for all experiment runs
AGENT_RUNS_DIR = "runs"

# Generate a log file
LOG_FILENAME = f"agent.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILENAME,
    filemode="w",
    force=True
)

# Constants for agent behavior
MODEL_NAME = "claude-3-7-sonnet-20250219"
MAX_RETRIES = 1
EXECUTION_TIMEOUT = 20
API_MAX_TOKENS = 2048
API_TEMPERATURE = 0.2

# System prompt for guiding AI code generation
SYSTEM_PROMPT = """
You are an expert Python software engineer specialized in generating code for scientific experiments.
Your goal is write correct, executable Python code based on the user's request. 

Instructions:
1. Respond with ONLY the required Python code
2. Enclose the entire Python code with a single ```python ... ``` markdown block
3. Do NOT include any explanations, introductory text, or concluding remarks outside the code block
4. Ensure the code is self contained or explicitly includes all necessary standard library imports.
5. The code MUST run non-interactively. Do not use `input()`.
6. If the experiment involves generating plots or files, save them to disk (e.g., `plot.png`, `results.csv`) rather than trying to display them directly.
7. If correcting previous code based on an error, provide the complete, corrected Python code block.
"""

REQ_SYSTEM_PROMPT = """
You are an expert Python dependency analyst. Given a Python script, your task is to identify the necessary pip installable packages and output them in a standard requirements.txt format

Instructions:
1. Analyze the import statements in the provided Python code.
2. Output ONLY the list of package names, one per line, exactly like a requirements.txt file
3. Do NOT include any version specifiers unless explicity requested
4. DO NOT include any python standard libraries
5. Do NOT include any explanations, introductory text, or concluding remarks
"""

def get_anthropic_client() -> anthropic.Anthropic:
    """
    Initializes and returns the Anthropic client using the API key from the 
    ANTHROPIC_API_KEY environment variable
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        logging.info("Anthropic client initialized successfully")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Anthropic client: {e}")
        raise
    
def extract_python_code(markdown: str) -> str | None:
    """
    Extracts Python code from markdown text
    """
    # Look for ```python ... ``` blocks
    python_pattern = r"```python\n(.*?)```"
    matches = re.findall(python_pattern, markdown, re.DOTALL)
    if matches:
        extracted = matches[0].strip()
        logging.info("Extracted Python code")
        return extracted
    logging.error("Failed to extract Python code")
    return None

def run_python_code(code: str, timeout: int, experiment_dir: str) -> tuple[int, str, str]:
    """
    Executes given Python code string in a temporary file using a subprocess
    
    Args:
        code: The Python code to execute
        timout: Maximum execution time in seconds
        
    Returns:
        A tuple (return_code, stdout, stderr):
        - return_code: The exit code of the subprocess
        - stdout: The standard output of the script
        - stderr: The standard error of the script or execution error message
    """
    script_name = "main.py"
    script_path = os.path.join(experiment_dir, script_name)
    start_time = time.time()
    captured_stdout_lines = []
    captured_stderr_lines = []
    
    try:
        # Write code to experiment.py
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(code)
            script_file.flush()
        logging.info(f"Executing generated code in file {script_path}")
        
        # Execute using same Python interpreter running this script
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=experiment_dir,
            bufsize=1 # Line-buffered
        )
        
        # Stream output until process finishes or times out
        while True:
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logging.error(f"Code execution timed out after {timeout} seconds")
                process.terminate()
                time.sleep(0.5)
                if process.poll() is None: # Check if still running
                    logging.warning("Process did not terminate, killing")
                    process.kill()
                return_code = -1
                break
            # Check if process finished
            if process.poll() is not None:
                break
            # Use select to wait for output on stdout or stderr
            ready_to_read, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1) 
            for stream in ready_to_read:
                if stream == process.stdout:
                    line = process.stdout.readline()
                    if line:
                        print(line, end='', flush=True) # Print to console in real-time
                        captured_stdout_lines.append(line)
                elif stream == process.stderr:
                    line = process.stderr.readline()
                    if line:
                        print(line, end='', file=sys.stderr, flush=True) # Print to console's stderr
                        captured_stderr_lines.append(line)
        
        stdout = "".join(captured_stdout_lines)
        stderr = "".join(captured_stderr_lines)
        logging.info(f"Execution finished with return code: {process.returncode}")
        return process.returncode, stdout, stderr
    
    # Handle exceptions
    except subprocess.TimeoutExpired:
        error_msg = f"Code execution timed out after {timeout} seconds."
        logging.error(error_msg)
        return -1, None, error_msg
    except FileNotFoundError:
        error_msg = f"File not found error during execution in {experiment_dir}. Interpreter: {sys.executable}"
        logging.error(error_msg)
        return -2, None, error_msg
    except Exception as e:
        error_msg = f"Failed to run subprocess: {e}"
        logging.exception(error_msg)
        return -1, None, error_msg

def install_requirements(requirements: str, timeout: int, experiment_dir: str) -> tuple[int, str, str]:
    """
    Install packages from a requirements string using pip
    
    Args: 
        requirements: string content of requirements.txt file
        timeout: max time in seconds for pip install command
    
    Returns: 
        A tuple (return_code, stdout, stderr) from the pip process
    """
    req_filename = "requirements.txt"
    req_path = os.path.join(experiment_dir, req_filename)
    stdout, stderr = "", ""
    try:
        # Write requirements to requirements.txt
        with open(req_path, "w", encoding="utf-8") as req_file:
            req_file.write(requirements)
            req_file.flush()
        logging.info(f"Installing requirements from file {req_path}")
        
        # Execute pip install command
        pip_cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
        logging.info(f"Running pip command: {' '.join(pip_cmd)}")
        process = subprocess.run(
            pip_cmd,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit
            timeout=timeout,
            encoding="utf-8"
        )
        
        stdout = process.stdout
        stderr = process.stderr
        logging.info(f"pip install finished. Return code: {process.returncode}")
        if process.returncode != 0:
            logging.error(f"pip install failed for {req_path}")
            logging.error(f"pip stderr:\n{stderr}")
            logging.error(f"pip stdout:\n{stdout}")
        return process.returncode, stdout, stderr
    
    except Exception as e:
        error_msg = f"Failed to install requirements: {e}"
        logging.exception(error_msg)
        if not stderr:
            stderr = error_msg
        return -1, stdout, stderr
    
class ExperimentAgent:
    """
    An AI agent that takes an experiment description, generates Python code using
    Claude, executes it, and iteratively refines the code based on execution errors
    until success or max retries is reached. 
    """
    def __init__(self):
        self.client = get_anthropic_client()
        self.experiment_dir = None
    
    def run_experiment(self, prompt: str) -> tuple[bool, str | None, str | None]:
        """
        Excecutes the full code cycle: generating, running, and refining on errors
        
        Args:
            prompt: Text description of experiment
        
        Returns:
            A tuple (success, final_code, output_msg)
            - success: True if code executed successfully, False otherwise
            - final_code: The last version of the code generated
            - output_msg: Stdout if successful, consolidated error message otherwise
        """
        # Ensure the parent agent_runs directory exists
        try:
            os.makedirs(AGENT_RUNS_DIR, exist_ok=True)
            logging.info(f"Ensured parent directory exists: {AGENT_RUNS_DIR}")
        except OSError as e:
             logging.error(f"Failed to create parent directory {AGENT_RUNS_DIR}: {e}")
             return False, None, f"Failed to create base directory {AGENT_RUNS_DIR}: {e}"
        # Create timestamped directory for this experiment run inside AGENT_RUNS_DIR
        timestamp = datetime.now().isoformat(timespec='seconds')
        experiment_folder_name = f"run-{timestamp}"
        # Store the full path
        self.experiment_dir = os.path.join(AGENT_RUNS_DIR, experiment_folder_name) 
        try:
            os.makedirs(self.experiment_dir, exist_ok=True) 
            logging.info(f"Created experiment directory: {self.experiment_dir}")
        except OSError as e:
             logging.error(f"Failed to create directory {self.experiment_dir}: {e}")
             return False, None, f"Failed to create directory {self.experiment_dir}: {e}"
        
        logging.info(f"Starting experiment")
        current_code = None
        error_context = None 
        last_stdout = None
        
        for attempt in range(MAX_RETRIES):
            logging.info(f"--- Attempt {attempt + 1} of {MAX_RETRIES} ---")
            
            # 1. Generate or refine code
            generated_code = self._generate_code(prompt, current_code, error_context)
            
            if not generated_code:
                fail_msg = "Failed to generate or extract code from API"
                logging.error(fail_msg)
                if error_context:
                    fail_msg += f"\nLast error context:\n{error_context}"
                return False, current_code, error_context
            current_code = generated_code
            logging.info("Generated code for current attempt")
            
            # 2. Generate and install requirements
            requirements = self._generate_requirements(current_code)
            if requirements:
                install_ret_code, install_stdout, install_stderr = install_requirements(
                    requirements, EXECUTION_TIMEOUT, self.experiment_dir
                )
                if install_ret_code == 0:
                    logging.info("Requirements installed successfully for this attempt")
                else:
                    # Failed to install requirements
                    logging.error("Failed to install generated requirements")
                    error_context = f"""
                        Failed to install generated requirements:
                        Pip Stderr:\n{install_stderr}\n
                        Pip Stdout:\n{install_stdout}\n
                        Generated requirements.txt:\n{requirements}\n
                    """
                    continue
            else:
                logging.error("Failed to generate requirements.txt content. Skipping attempt")
                continue
            
            # 3. Execute generated code
            logging.info("Running generated code")
            return_code, stdout, stderr = run_python_code(
                current_code, EXECUTION_TIMEOUT, self.experiment_dir
            )
            last_stdout = stdout
            
            # 4. Evaluate execution result
            if return_code == 0: 
                # Success
                logging.info("Experiment executed successfully")
                if stderr:
                    # Log warnings even on success
                    logging.warning(f"Execution successful, but stderrr contained:\n{stderr}")
                return True, current_code, stdout
            
            # Failure: prepare error context for next generation attempt
            error_context = f"Code execution failed with return code {return_code}.\n"
            logging.warning(error_context)
            if stderr:
                error_context += f"Stderr:\n{stderr}\n"
            else:
                error_context += "No output on stderr\n"
            if stdout:
                error_context += f"Stdout:\n{stdout}\n"
            logging.info("Attempting to refine code based on error...")
        
        # Max retries reached
        logging.error(f"Experiment failed after {MAX_RETRIES} attempts")
        fail_msg = f"Failed after {MAX_RETRIES} attempts"
        if error_context:
            fail_msg += f"\nLast error:\n{error_context}"
        if last_stdout:
            fail_msg = f"\nLast stdout:\n{last_stdout}"
        return False, current_code, fail_msg
        
    def _generate_code(self, prompt: str, previous_code: str, last_error: str) -> str | None:
        """
        Generates Python code for given experiment using Claude
        
        Args:
            prompt: The original experiment description
            previous code: The code from the previous attempt (if any)
            error context: The error message and context from the previous failed
                           execution (if any)
                           
        Returns: 
            The extracted Python code string, or None if API call or execution fails
        """
        if not previous_code:
            # Initial code generation request
            user_content = f"""
                Generate Python code to implement the following experiment:
                {prompt}
                Remember to only output the code within a ```python block
            """
        else:
            # Code refinement request based on error
            user_content = f"""
                The following code, intended to implement the experiment described below, produced an error:
                Original Experiment:\n{prompt}\n\n
                Code Attempt:\n{previous_code}\n\n
                Error:\n{last_error}\n\n
                Please refine the code to fix the error. Provide the complete, corrected Python code block only. 
            """
        messages = [{"role": "user", "content": user_content}]
        
        try:
            # Call Anthropic API
            response = self.client.messages.create(
                model=MODEL_NAME,
                messages=messages,
                system=SYSTEM_PROMPT,
                max_tokens=API_MAX_TOKENS,
                temperature=API_TEMPERATURE
            )
            logging.info("Received response from Anthropic API")
            
            # Extract text from response
            response_text = ""
            if response.content and isinstance(response.content, list):
                 # Assume code is in the first block
                response_text = response.content[0].text
            else:
                logging.warning(f"Unexpected response structure: {response}")
                # Attempt to convert to string as fallback
                response_text = str(response.content)
                
            # Extract Python code from text
            extracted_code = extract_python_code(response_text)
            if not extracted_code:
                logging.error(f"Could not extract Python code from API response:\n{response_text}")
                return None
            return extracted_code

        # Handle API errors and exceptions
        except anthropic.APIConnectionError as e:
            logging.error(f"Anthropic API connection error: {e}")
        except anthropic.RateLimitError as e:
            logging.error(f"Anthropic API rate limit exceeded: {e}")
        except anthropic.APIStatusError as e:
            logging.error(f"Anthropic API status error: {e.status_code} - {e.response}")
        except Exception as e:
            logging.exception(f"Anthropic API call failed unexpectedly: {e}")
        return None
    
    def _generate_requirements(self, code: str) -> str | None:
        """
        Asks LLM to generate requirements.txt content based on the code provided
        """
        user_content = f"""
            Based only on the import statements in the following Python code, generate a requirements.txt listing the necessary pip packages:
            \n{code}\n
            Output ONLY the requirements.txt content, one package per line. 
        """
        messages = [{"role": "user", "content": user_content}]
        try:
            logging.info("Generating requirements.txt via API")
            response = self.client.messages.create(
                model=MODEL_NAME,
                max_tokens=512,
                messages=messages,
                system=REQ_SYSTEM_PROMPT,
                temperature=0.0
            )
            response_text = ""
            if response.content and isinstance(response.content, list):
                 # Assume code is in the first block
                response_text = response.content[0].text
            else:
                logging.warning(f"Unexpected response structure: {response}")
                # Attempt to convert to string as fallback
                response_text = str(response.content)
            logging.info("Recieved requirements response from API")
            return response_text
        except Exception as e:
            logging.exception(f"Failed to generate requirements via API: {e}")
            return None

if __name__ == "__main__":
    # Get prompt
    prompt = "Generate a non-linear data and train a neural network on it as best as you can."
    
    # Log the start and the log file being used
    print(f"--- Experiment Agent Started ---")
    print(f"Logging output to: {LOG_FILENAME}")
    logging.info("\n" + "="*10 + " RUNNING EXPERIMENT AGENT " + "="*10 + "\n")
    
    final_report = ""
    try:
        # Run agent
        agent = ExperimentAgent()
        success, final_code, result_output = agent.run_experiment(prompt)
        
        # Format final report
        final_report += "\n" + "="*20 + " EXPERIMENT REPORT " + "="*20 + "\n\n"
        final_report += f"STATUS: {'Success' if success else 'Failed'}" + "\n\n"
        final_report += "="*10 + " EXECUTED CODE " + "="*10 + "\n\n"
        if final_code:
            final_report += final_code + "\n"
        else:
            final_report += "No code successfully executed" + "\n\n"
        final_report += "\n" + "="*10 + " OUTPUT " + "="*10 + "\n\n"
        if success:
            final_report += (result_output if result_output else "No standard output")
        else:
            final_report += (result_output if result_output else "No specific error message")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        # Log the final report string to the file
        logging.info(final_report)
        print("Agent finished. Check log file for details")
    