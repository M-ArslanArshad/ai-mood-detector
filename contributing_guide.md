# Contributing to AI Mood Detector

Thank you for considering contributing to the AI Mood Detector project! We welcome contributions from everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

We love to receive enhancement suggestions! Please create an issue with:
- Clear description of the enhancement
- Why this would be useful
- Possible implementation approach

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/YourFeatureName
   ```

2. **Make your changes**
   - Write clean, readable code
   - Follow PEP 8 style guidelines
   - Add comments where necessary
   - Update documentation if needed

3. **Test your changes**
   - Ensure all existing tests pass
   - Add new tests for new features
   - Test on different audio samples

4. **Commit your changes**
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```
   
   Use prefixes:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Docs:` for documentation changes

5. **Push to your fork**
   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Open a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Explain what you changed and why

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small
- Use type hints where appropriate

Example:
```python
def extract_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Extract MFCC features from audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Sampling rate (default: 16000)
        
    Returns:
        Extracted MFCC features as numpy array
    """
    # Implementation
    pass
```

## Development Setup

1. Clone your fork
   ```bash
   git clone https://github.com/your-username/ai-mood-detector.git
   ```

2. Create virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # for development tools
   ```

4. Create a new branch
   ```bash
   git checkout -b feature/your-feature
   ```

## Testing

Before submitting a PR:
- Test with various audio files
- Check for memory leaks with large files
- Verify the Streamlit app works correctly
- Ensure training script runs without errors

## Documentation

- Update README.md if you change functionality
- Add docstrings to new functions
- Update comments if you modify existing code
- Include examples for new features

## Areas for Contribution

We especially welcome contributions in:
- **Model Improvements**: Better architectures, hyperparameter tuning
- **Dataset Support**: Adding new emotion datasets
- **Performance**: Optimization for faster inference
- **Features**: Real-time audio recording, batch processing
- **UI/UX**: Improving the Streamlit interface
- **Documentation**: Better examples, tutorials, guides
- **Testing**: Unit tests, integration tests
- **Multilingual Support**: Non-English emotion recognition

## Questions?

Feel free to reach out to any of the project authors:
- Ghulam Ahmad: 2021ee305@uet.edu.pk
- Zeehsan Anjum: 2021ee299@uet.edu.pk
- Ahmed Shahfique: 2021ee313@uet.edu.pk
- Arslan Arshad: 2021ee314@uet.edu.pk

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards other community members

Thank you for contributing! 🎉
