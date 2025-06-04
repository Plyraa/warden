@echo off
echo 🚀 Testing Streaming Batch Endpoint with curl
echo ================================================
echo.
echo Files to process:
echo   - test1.mp3
echo   - 16952962156823ffe2a06954.22932860.mp3
echo   - 66057328368247d623d0b87.67876133.mp3
echo   - 2097202462683d9152e1d734.24483919.mp3
echo   - 243801406824559add7684.37683750.mp3
echo.
echo ⏱️  Results will stream in real-time...
echo ================================================
echo.

curl -X POST "http://localhost:8000/batch-stream" ^
  -H "Content-Type: application/json" ^
  -H "Accept: application/x-ndjson" ^
  --no-buffer ^
  -w "\n\n⏱️  Total time: %%{time_total}s\n📊 Response code: %%{response_code}\n" ^
  -d "{\"file_paths\": [\"C:\\Users\\Plyra\\Desktop\\Plyra\\jotform\\warden\\stereo_test_calls\\test1.mp3\", \"C:\\Users\\Plyra\\Desktop\\Plyra\\jotform\\warden\\stereo_test_calls\\16952962156823ffe2a06954.22932860.mp3\", \"C:\\Users\\Plyra\\Desktop\\Plyra\\jotform\\warden\\stereo_test_calls\\66057328368247d623d0b87.67876133.mp3\", \"C:\\Users\\Plyra\\Desktop\\Plyra\\jotform\\warden\\stereo_test_calls\\17457745626800ad609b0bd7.58327851.mp3\", \"C:\\Users\\Plyra\\Desktop\\Plyra\\jotform\\warden\\stereo_test_calls\\243801406824559add7684.37683750.mp3\"]}"

echo.
echo ✅ Streaming test complete!
pause
