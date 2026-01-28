import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from agent.model_wrappers import TinyLlamaWrapper, Qwen2Wrapper

def test():
    # A simple schema for testing
    schema = "CREATE TABLE students (id INT, name TEXT, age INT);"
    question = "Count how many students are older than 20."

    print("\n" + "="*40)
    print("🤖 MODEL TEST 1: TinyLlama (1.1B Chat)")
    print("="*40)
    try:
        tiny = TinyLlamaWrapper()
        print(f"👉 INPUT: {question}")
        print(f"✅ OUTPUT: {tiny.generate_sql(question, schema)}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    print("\n" + "="*40)
    print("🤖 MODEL TEST 3: Qwen2.5-1.5B-Instruct")
    print("="*40)
    try:
        qwen = Qwen2Wrapper()
        print(f"👉 INPUT: {question}")
        print(f"✅ OUTPUT: {qwen.generate_sql(question, schema)}")
    except Exception as e:
        print(f"❌ FAILED: {e}")


if __name__ == "__main__":
    test()