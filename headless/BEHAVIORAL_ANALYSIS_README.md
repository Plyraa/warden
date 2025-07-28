# Behavioral Analysis Features

This document describes the new behavioral analysis features added to the Warden audio analysis system.

## Overview

Two new analysis cases have been implemented to detect:

1. **Customer Churn Risk**: Identifies when customers show genuine anger and intent to stop using the product/service
2. **Repetition Detection**: Identifies problematic repetitive patterns in both user and agent communication

## Implementation

### Technology Stack
- **Gemini 2.5 Pro**: Multimodal AI for audio analysis
- **Structured Output**: Pydantic models for consistent response format
- **Proxy Integration**: Uses company-provided Gemini API proxy

### New Components

#### 1. GeminiBehavioralAnalyzer (`gemini_behavioral_analyzer.py`)
- Core behavioral analysis engine
- Uses Gemini 2.5 Pro multimodal processing
- Structured output with Pydantic validation
- Batch processing capabilities

#### 2. Updated Schemas (`schemas.py`)
New response fields added:
```python
userChurnRisk: Optional[bool]           # Customer churn risk flag
userChurnReasoning: Optional[str]       # 1-2 sentence explanation
userRepetition: Optional[bool]          # User repetition flag  
agentRepetition: Optional[bool]         # Agent repetition flag
```

#### 3. Enhanced Audio Processor (`audio_processor.py`)
- Optional behavioral analysis integration
- Configurable via `enable_behavioral_analysis` parameter
- Seamless integration with existing metrics

#### 4. API Endpoints (`app.py`)
- Optional behavioral analysis via query parameter
- Two service instances (with/without behavioral analysis)
- Backward compatibility maintained

#### 5. Fast CSV Logger (`fast_behavioral_csv_logger.py`)
- Dedicated tool for behavioral analysis only
- Faster processing compared to full analysis
- CSV output with behavioral metrics only

## Usage

### API Integration

#### Enable behavioral analysis via query parameter:
```bash
POST /batch?behavioral_analysis=true
POST /batch-stream?behavioral_analysis=true
```

#### Response includes new fields:
```json
{
  "userChurnRisk": true,
  "userChurnReasoning": "Customer repeatedly called the bot useless and threatened to cancel service.",
  "userRepetition": false,
  "agentRepetition": true
}
```

### Standalone Behavioral Analysis

#### Run fast behavioral analysis only:
```bash
python fast_behavioral_csv_logger.py
```

#### Test the implementation:
```bash
python test_behavioral_analysis.py
```

## Analysis Criteria

### Case 1: Customer Churn Risk

**TRUE indicators (churn risk):**
- Customer explicitly states they will stop using the product/service
- Customer swears at or insults the agent/bot ("bad", "useless", "stupid")
- Customer expresses extreme frustration with the product itself
- Customer threatens to leave or find alternatives
- Customer shows sustained anger about product quality

**FALSE indicators (not churn risk):**
- Technical workflow failures or temporary issues
- Minor frustration that gets resolved
- Complaints about specific features without intent to leave
- Neutral feedback or constructive criticism

### Case 2: Repetition Detection

#### Agent Repetition
**TRUE indicators (problematic):**
- Agent repeats the same greeting multiple times without reason
- Agent gives identical responses to different user inputs
- Agent loops through the same script without adapting
- Agent provides the same unhelpful response repeatedly

**FALSE indicators (acceptable):**
- Confirmation repetitions for verification (email, phone, etc.)
- Workflow-required repetitions
- Clarification repetitions when user didn't understand

#### User Repetition
**TRUE indicators (problematic):**
- User repeatedly asks for the same reasonable thing that agent doesn't address
- User rephrases the same request multiple times without getting help
- User continuously repeats their goal because agent misunderstands
- User shows frustration from having to repeat themselves

**FALSE indicators (acceptable):**
- Workflow-required repetitions
- User correcting their own mistakes
- User providing additional details upon request

## Configuration

### Environment Variables
```bash
PROXY_API_KEY="your-proxy-api-key"
GEMINI_PROXY_BASE_URL="https://dev.jotform.ai/gemini/v1beta/models"
```

### File Configuration
Update paths in configuration files:
- `fast_behavioral_csv_logger.py`: Set `AUDIO_FILES_DIR` and `INPUT_CSV`
- `test_behavioral_analysis.py`: Update test file paths

## Performance Considerations

- Behavioral analysis adds ~2-3 seconds per file
- Uses separate service instances to avoid impacting standard processing
- Optional feature - can be disabled for faster processing
- Batch processing includes API rate limiting (2-second delays)

## Output Files

### Standard CSV Logger
Includes all metrics plus new behavioral fields

### Fast Behavioral CSV Logger
Focused output with behavioral metrics only:
- `userChurnRisk`
- `userChurnReasoning` 
- `userRepetition`
- `agentRepetition`
- `processing_time_seconds`

## Error Handling

- Graceful fallback when behavioral analysis fails
- Existing metrics continue to work even if behavioral analysis is unavailable
- Detailed error logging for debugging
- Optional nature ensures system stability

## Testing

Run the test suite to verify functionality:
```bash
python test_behavioral_analysis.py
```

The test suite verifies:
1. Gemini behavioral analyzer initialization
2. Audio processor integration
3. Fast CSV logger functionality
4. Error handling and edge cases
