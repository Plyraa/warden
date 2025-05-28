import time
import requests

test_url = (
    "https://github.com/Plyraa/warden/raw/refs/heads/main/stereo_test_calls/test1.mp3"
)

print("Testing URL reuse with timing...")

# First request - should download and process
print("Making first request...")
start1 = time.time()
response1 = requests.post(
    "http://localhost:8000/batch", json={"file_paths": [test_url]}
)
end1 = time.time()
time1 = end1 - start1

print(f"First request took: {time1:.2f} seconds")
print(f"First request status: {response1.status_code}")

# Second request - should reuse existing
print("Making second request...")
start2 = time.time()
response2 = requests.post(
    "http://localhost:8000/batch", json={"file_paths": [test_url]}
)
end2 = time.time()
time2 = end2 - start2

print(f"Second request took: {time2:.2f} seconds")
print(f"Second request status: {response2.status_code}")

speedup = time1 / time2 if time2 > 0 else float("inf")
print(f"Speedup: {speedup:.1f}x faster")

if time2 < time1 * 0.5:  # If second request is at least 2x faster
    print("✅ SUCCESS: Significant speedup indicates URL reuse is working!")
else:
    print("⚠️  May not be reusing - similar timing between requests")
