import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

# Setup ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.maximize_window()

# Open WhatsApp Web
driver.get('https://web.whatsapp.com/')
print("Scan the QR code with your WhatsApp mobile app to login")

# Wait for QR code scan and page load
WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="pane-side"]'))
)
print("Logged in successfully!")

# Locate and click the first chat
try:
    first_chat = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="pane-side"]/div/div/div[1]/div/div/div[1]'))
    )
    first_chat.click()
    print("Clicked on the first chat.")
except Exception as e:
    print("Error clicking the chat:", e)
    driver.quit()
    exit()

# Wait for the chat to open and locate messages
try:
    messages = WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.XPATH, '//*[@id="main"]/div[3]/div/div/div[3]/div'))
    )
    for msg in messages[-2:]:
        print("Message:")
        print(msg.text)
except Exception as e:
    print("Error retrieving messages:", e)

# Locate the message input box and send a test message
try:
    message_box = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/footer/div[1]/div/span/div/div[2]/div[1]/div[2]'))
    )
    message_box.click()  # Ensure the input box is active
    message_box.send_keys('This is a test message')
    message_box.send_keys(Keys.RETURN)
    print("Message sent successfully!")
except Exception as e:
    print("Error sending message:", e)

# Wait to observe results
time.sleep(5)
driver.quit()
