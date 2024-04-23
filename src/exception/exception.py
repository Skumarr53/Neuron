import sys

class DetailedError(Exception):
    def __init__(self, message, _sys: sys=sys):
        self.message = message
        # Capture the current stack trace
        stack = _sys.exc_info()[2]  # Get the second last entry to ignore the current __init__ call
        self.script_name = stack.tb_frame.f_code.co_filename
        self.line_number = stack.tb_lineno
    
    def __str__(self):
        return f"Error occurred in {self.script_name} \nat line: {self.line_number}\nError Message: {self.message}"

# Example usage
if __name__ == "__main__":
  try:
      a = 1/0
  except Exception as e:
      raise DetailedError(e)