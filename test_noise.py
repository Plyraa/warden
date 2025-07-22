import os
from noise_detection import detect_noise, NOISE_TEST_DIR

def run_noise_detection_tests():
    """
    Tests the noise detection functionality on audio files in the specified directory.
    """
    if not os.path.exists(NOISE_TEST_DIR):
        print(f"Error: Test directory '{NOISE_TEST_DIR}' not found.")
        return

    print("-" * 80)
    print(f"Running noise detection tests on files in '{NOISE_TEST_DIR}'...")
    print("-" * 80)

    results = []
    for filename in sorted(os.listdir(NOISE_TEST_DIR)):
        if filename.lower().endswith(('.wav', '.mp3')):
            file_path = os.path.join(NOISE_TEST_DIR, filename)
            print(f"Testing file: {filename}")
            res = detect_noise(file_path)
            results.append(res)
            print(f"  -> Result: {res}")


    print("\n" + "=" * 80)
    print(" " * 28 + "NOISE DETECTION RESULTS")
    print("=" * 80)
    print(f"{'File Name':<40} {'Has Noise':<12} {'Noise Interrupt':<16} {'Error'}")
    print("-" * 80)

    for res in results:
        status = "Yes" if res.get("hasNoise") else "No"
        interrupt_status = "Yes" if res.get("noiseInterrupt") else "No"
        details = res.get("error", "N/A")
        
        # Highlight noisy files for easier review
        if res.get("hasNoise"):
            print(f"\033[93m{res['file']:<40} {status:<12} {interrupt_status:<16} {details}\033[0m")
        else:
            print(f"{res['file']:<40} {status:<12} {interrupt_status:<16} {details}")

    print("-" * 80)
    print("Test run complete.")

if __name__ == "__main__":
    run_noise_detection_tests()
