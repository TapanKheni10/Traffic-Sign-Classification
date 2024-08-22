## Traffic-Sign-Classification 🚦

In recent years, we've witnessed a remarkable surge in autonomous vehicle technology, with major automotive companies vying to develop highly reliable and efficient self-driving cars. These innovative vehicles face numerous challenges, one of the most critical being accurate traffic sign detection. This capability is essential for ensuring compliance with traffic rules and regulations, thereby preventing potentially dangerous situations on the road.

Here's a more polished and fluent version of your text:
In recent years, we've witnessed a remarkable surge in autonomous vehicle technology, with major automotive companies vying to develop highly reliable and efficient self-driving cars. These innovative vehicles face numerous challenges, one of the most critical being accurate traffic sign detection. This capability is essential for ensuring compliance with traffic rules and regulations, thereby preventing potentially dangerous situations on the road.

My project represents a focused approach to addressing this crucial aspect of autonomous driving through the application of Convolutional Neural Networks (CNNs). Leveraging the power of the PyTorch framework, I've developed a model capable of classifying a wide array of traffic signs with an impressive accuracy rate exceeding 90%. I have also used OpenCV to make it work in the real-time.

This high level of accuracy is vital for real-world applications, as it significantly enhances the ability of autonomous vehicles to interpret and respond to traffic signage correctly. By contributing to the advancement of traffic sign recognition technology, this project aims to play a part in improving the overall safety and reliability of self-driving systems.

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Technologies">Technologies</a>
    </li>
    <li>
	    <a href = "#How-to-Start-Project">How to Start Project</a>
    </li>
    <li>
	    <a href = "#Contact-Us">Contact Us</a>
    </li>
    <li>
	    <a href = "#License">License</a>
    </li>
  </ol>
</details>

## Technologies
- Python
- OpenCV
- PyTorch

## How to Start Project
Follow these steps to get started with the project:

1. **Clone the Repository:**
   ```bash
   git clone <repository_link>
   ```
2. **Install Anaconda:**
   
   Make sure you have Anaconda installed on your system. If not, you can download and install it from the official website: https://www.anaconda.com/download/
   
4. **Create a Virtual Environment:**
   
   Create a new virtual environment using Python 3.9.6:

   ```bash
   conda create --name your_env_name python=3.9.6 -y
   ```
   Replace your_env_name with the desired name for your virtual environment.
   
   Activate the newly created environment:
   ```bash
   conda activate your_env_name
   ```
5. **Install Dependencies:**
   
   Install the project dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
   This command will install all the required packages listed in the requirements.txt file.

7. **Run the main.py file:**
   ```bash
   python main.py
   ```

## Contact Us
To learn more about our system and how it can help to reduce cost, please reach out:

📧 tapankheni10304@gmail.com

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Note:
You might face certain issues related to numpy version because torchvision uses numpy behind the scene to deal with the image data and sometimes it causes version compatibility issues.
Ever get stuck, just run the following command:
```bash
   pip install --upgrade numpy torchvision
```