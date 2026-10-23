# ParkMate

### Requirements
- Python 3.8 or above
- OpenCV (`opencv-python`)
- Python standard libraries: `json`, `os`, and `sys`
- Parking slot configuration files: `slots.json` and `demo_slots.json`
- Input parking lot image or video for vehicle occupancy detection

Install the required library using:

```bash
pip install opencv-python
```



### Sample Execution

1. Clone or download the ParkMate repository from GitHub.
2. Open the project folder in VS Code.
3. Install the required dependency:

```bash
pip install opencv-python
```

4. Navigate to the `ParkMate/buddhi` folder.
5. Run the parking occupancy detection script:

```bash
python detect_occupancy.py
```

To check the occupancy status, run:

```bash
python check_occupancy.py
```

To select parking slots, run:

```bash
python select_slots.py
```

**Expected Output:** The system detects parking slot occupancy and displays or saves the parking occupancy results, depending on the script configuration.
