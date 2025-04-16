import ollama
import pandas as pd
import re
import hashlib
import json
import subprocess
import tempfile
import os
import time
from tqdm import tqdm

class AutoMLGenerator:
    def __init__(self):
        self.max_retries = 2
        self.start_time = time.time()

        pass

    def _log(self, message: str):
        """Show timestamped progress updates"""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:5.1f}s] {message}")

    def _error(self, message: str):
        """Format error messages clearly"""
        print(f"\n{'!'*50}\nERROR: {message}\n{'!'*50}\n")

    def generate_ml_pipeline(self, csv_path: str, user_query: str) -> dict:
        """Main pipeline generation workflow"""
        try:
            self._log("Analyzing dataset...")
            df = pd.read_csv(csv_path)
            data_summary = self._analyze_data(df)
            
            with tqdm(total=self.max_retries+1, desc="Progress") as pbar:
                for attempt in range(self.max_retries + 1):
                    # Generate/correct code with strict formatting
                    if attempt == 0:
                        code = self._generate_initial_code(data_summary, user_query, df, csv_path)
                    else:
                        code = self._correct_code(code, error_output, df, csv_path)
                    
                    # Clean and validate code
                    clean_code = self._clean_and_validate_code(code, csv_path)
                    validation_result = self._execute_code_test(clean_code, csv_path)
                    
                    if validation_result['status'] == 'success':
                        return self._save_success(clean_code, csv_path, user_query, data_summary, attempt+1)
                    
                    error_output = validation_result['error']
                    self._error(f"Attempt {attempt+1} failed: {error_output}")
                    pbar.update(1)
                
                raise RuntimeError(f"Failed after {self.max_retries+1} attempts")

        except Exception as e:
            return self._handle_error(e)

    def _generate_initial_code(self, data_summary: dict, user_query: str, df: pd.DataFrame, csv_path: str) -> str:
        """Generate initial code version with explicit path"""
        self._log("Generating initial code...")
        response = ollama.generate(
            model="llama3",
            prompt=f"""Generate Python code that:
            1. Loads data from: {csv_path}
            2. Processes columns: {data_summary['columns']}
            3. Predicts: {data_summary['target']}
            4. Requirements: {user_query}
            
            Must include:
            import pandas as pd
            data = pd.read_csv(r"{csv_path}")
            
            Respond ONLY with code between ```python and ```""",
            options={"temperature": 0.2}
        )
        return response['response']

    def _correct_code(self, code: str, error: str, df: pd.DataFrame, csv_path: str) -> str:
        """Generate corrected code version with path validation"""
        self._log("Correcting code...")
        response = ollama.generate(
            model="llama3",
            prompt=f"""Fix this code that failed with: {error}
            Original CSV path: {csv_path}
            Must use: pd.read_csv(r"{csv_path}")
            Code to fix:
            {code}
            Respond ONLY with corrected code between ```python and ```""",
            options={"temperature": 0.1}
        )
        return response['response']

    def _clean_and_validate_code(self, raw_code: str, csv_path: str) -> str:
        """Clean and validate generated code"""
        # Extract code block from markdown
        code_blocks = re.findall(r'```python\n(.*?)\n```', raw_code, re.DOTALL)
        clean_code = '\n'.join(code_blocks).strip() if code_blocks else raw_code
        
        # Ensure CSV path is correct
        clean_code = clean_code.replace('data.csv', f'r"{csv_path}"')
        clean_code = clean_code.replace("pd.read_csv('data.csv')", f'pd.read_csv(r"{csv_path}")')
        
        # Validate critical patterns
        if f'pd.read_csv(r"{csv_path}")' not in clean_code:
            raise ValueError("CSV path not properly set in generated code")
            
        return clean_code

    def _execute_code_test(self, code: str, csv_path: str) -> dict:
        """Execute code in isolated environment with proper paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "temp_script.py")
            with open(script_path, 'w') as f:
                f.write(code)
            
            try:
                result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(csv_path),  # Run in CSV's directory
                    timeout=120
                )
                
                return {
                    'status': 'success' if result.returncode == 0 else 'error',
                    'error': f"{result.stderr}\n{'-'*50}\n{result.stdout}"
                }
            
            except subprocess.TimeoutExpired:
                return {'status': 'error', 'error': "Timeout after 2 minutes"}

    def _analyze_data(self, df: pd.DataFrame) -> dict:
        """Analyze dataset structure"""
        return {
            'columns': list(df.columns),
            'dtypes': dict(df.dtypes.apply(lambda x: str(x))),
            'target': df.columns[-1],
            'samples': len(df),
            'missing': df.isna().sum().to_dict()
        }

    def _save_success(self, code: str, csv_path: str, query: str, analysis: dict, attempts: int) -> dict:
        """Save successful pipeline"""
        filename = f"pipeline_{hashlib.md5((csv_path + query).encode()).hexdigest()}.py"
        with open(filename, 'w') as f:
            f.write(code)
        
        self._log(f"Success in {attempts} attempts!")
        return {
            'status': 'success',
            'code_path': filename,
            'analysis': analysis,
            'attempts': attempts,
            'time': time.time() - self.start_time
        }

    def _handle_error(self, error: Exception) -> dict:
        """Format error response"""
        return {
            'status': 'error',
            'message': str(error),
            'time': time.time() - self.start_time
        }

if __name__ == "__main__":
    print("🚀 Robust ML Pipeline Generator")
    generator = AutoMLGenerator()
    
    # csv_path = r"D:\Applied AI\Sem-2\Full Stack data science\AI-Co-Pilot\.csv"
    csv_path = r"D:\Applied AI\Sem-2\Full Stack data science\AI-Co-Pilot\CVD_cleaned.csv"
    result = generator.generate_ml_pipeline(
        csv_path=csv_path,
        user_query="Build a machine learning prediction model to identify the likelihood of skin cancer in males based on their age and weight. Use appropriate preprocessing, select the best algorithm for the task, and evaluate the model's performance."
        # user_query="Build a sentiment analysis model with text input"
    )
    
    if result['status'] == 'success':
        print(f"\n✅ Success in {result['time']:.1f}s")
        print(f"📁 File: {result['code_path']}")
        print(f"🔄 Attempts: {result['attempts']}")
    else:
        print(f"\n❌ Failed after {result.get('attempts', 0)} attempts")
        print(f"Error: {result['message']}")