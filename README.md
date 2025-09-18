# Aspect-Wizard

*(Add your app description here)*

---

## Prerequisites

Before installing, make sure you have:

- **Python 3.10+**
- **Git**

Check with:
```bash
git --version
python --version
```

---

## If you do **NOT** have Git or Python

1. **Download this repo as ZIP**  
   Go to [Aspect-Wizard GitHub](https://github.com/alihassan-ai/Aspect-Wizard), click the green **Code** button, then choose **Download ZIP**. Extract it.

2. **Install Git and Python**  
   - [Download Git](https://git-scm.com/downloads)  
   - [Download Python 3.10+](https://www.python.org/downloads/)

3. **Run installer**  
   In the extracted folder, double-click:
   ```
   install.bat
   ```
   (or run from Command Prompt: `.\install.bat`)

4. **Launch the app**  
   ```
   launch.bat
   ```

---

## If you already have Git and Python installed

1. **Clone the repo**
   ```bash
   git clone https://github.com/alihassan-ai/Aspect-Wizard.git
   cd Aspect-Wizard
   ```

2. **Run installer**
   ```bash
   install.bat
   ```

3. **Launch the app**
   ```bash
   launch.bat
   ```

---

## Troubleshooting

- If `git` or `python` commands are not recognized, install them and ensure they are added to your PATH.  
- If the installer fails, delete the `venv` folder and re-run `install.bat`.  
- For manual setup (Linux/macOS):  
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  python app.py   # or your actual entry point
  ```

---

## License

(Add license info here)
