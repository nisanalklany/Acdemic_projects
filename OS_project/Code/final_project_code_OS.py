import sys 
import os 
#import subprocess
import multiprocessing  # Importing multiprocessing module for parallel processing
from multiprocessing import Manager  # Importing Manager for shared resources between processes
import threading  # Importing threading module for parallel execution of tasks
import time  # Importing time module for performance measurement
import math  # Importing math module for mathematical operations
import numpy as np  # Importing numpy for numerical operations
from multiprocessing import Manager  # Importing Manager again (redundant)

# Creating a threading lock to synchronize access to shared resources
lock_threading = threading.Lock()
num_of_subject = 0  # Counter for the number of subjects
subjects_lists = []  # List to store subjects

class Subject:
    def __init__(self, name, scores, data_path):
        self.name = name  # Subject name
        self.scores = scores  # List of scores for the subject
        self.lock_process = multiprocessing.Lock()  # Lock for process synchronization
        self.data_path = data_path  # Path to store output data
        self.mean = None  # Variable to store mean of scores
        self.median = None  # Variable to store median of scores
        
   

    def calculate_stats(self, num_of_process, queue):
        if not self.scores:  # Check if there are scores available
            print(f"Warning: No scores found for {self.name}")
            return
        
        self.mean = np.mean(self.scores)  # Calculate mean of scores
        self.median = np.median(self.scores)  # Calculate median of scores
        #self.process_subjects_in_parallel(num_of_process, queue)  # Start parallel processing
        
        histogram_all = np.zeros(5)  # Initialize histogram array
        while not queue.empty():  # Retrieve histogram results from queue
            histogram_all += np.array(queue.get())
             
        total_scores = len(self.scores)  # Get total number of scores
        percentage_scores = (histogram_all / total_scores) * 100  # Convert histogram to percentage
        self.write_data(percentage_scores)  # Write results to file

    def histogram(self, scores_list, queue):
        # Create a histogram of scores with predefined bins
        histogram, bin_edges = np.histogram(scores_list, bins=[0, 40, 55, 70, 85, 101])
        queue.put(histogram)  # Store histogram in queue
        
    def write_data(self, percentage_scores):
        total_grades = len(self.scores)  # Total number of grades
        print(f"Subject: {self.name}")
        print(f"Mean: {self.mean}")
        print(f"Median: {self.median}")
        
        # Define summary file path
        summary_file_path = os.path.join(self.data_path, "grades_summary.txt")
        
        # Prepare summary content
        main_content = f"""
            Domain: {self.name}
            Total Grades: {total_grades}
            Mean: {round(self.mean, 2)}
            Median: {round(self.median, 2)}
            Histogram:
            0-39: {round(percentage_scores[0], 2)}%
            40-54: {round(percentage_scores[1], 2)}%
            55-69: {round(percentage_scores[2], 2)}%
            70-84: {round(percentage_scores[3], 2)}%
            85-100: {round(percentage_scores[4], 2)}%
              """
        
        self.lock_process.acquire()  # Lock process to ensure safe file writing
        with open(summary_file_path, "a") as file:
            file.write(main_content)  # Write summary to file
        self.lock_process.release()  # Release lock
        print(f"Summary written for subject: {self.name}")

# Function to read scores from a file and store in Subject object
#  def process_subjects_in_parallel( num_of_process, queue):
#         processes = []  # List to store process instances
#         chunk_size = len(scores) // num_of_process  # Determine chunk size for each process
        
#         # Creating and starting multiple processes for histogram calculation
#         for proc in range(num_of_process):
#             start_idx = proc * chunk_size  # Start index for chunk
#             end_idx = len(scores) if proc + 1 == num_of_process else (proc + 1) * chunk_size  # End index for chunk
#             p = multiprocessing.Process(target=self.histogram, args=(self.scores[start_idx:end_idx], queue))  # Creating process
#             processes.append(p)  # Adding process to list
#             p.start()  # Starting process
        
#         # Waiting for all processes to complete
#         for p in processes:
#             p.join()


def read_and_save(file_name, data_path): 
    global subjects_lists  # Access global subjects list
    scores = []  # List to store scores
    subject_name = file_name.replace(".txt", "")  # Extract subject name from file name
    
    # Read scores from file
    with open(file_name, "r") as file:
        for line in file:
            scores.append(float(line.strip()))  # Convert score to float and store
    
    lock_threading.acquire()  # Acquire lock for thread safety
    subjects_lists.append(Subject(subject_name, scores, data_path))  # Append new Subject object to list
    lock_threading.release()  # Release lock
    
if __name__ == "__main__":
    num_cores = os.cpu_count()  # Get number of CPU cores
    queue = multiprocessing.Queue()  # Create a queue for inter-process communication
    current_directory = os.getcwd()  # Get current working directory
    print(f"The current directory is: {current_directory}")
    
    # Define data directory path
    data_path = os.path.join(current_directory, "data")
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f"Data directory not found at: {data_path}")
        sys.exit(1)  # Exit if data directory does not exist
    
    os.chdir(data_path)  # Change working directory to data folder
    print(f"The directory changed to {os.getcwd()}")
    
    # Clean summary file if it exists
    summary_file_path = os.path.join(data_path, "grades_summary.txt")
    if os.path.exists(summary_file_path):
        os.remove(summary_file_path)  # Remove existing summary file
        print("Cleared existing summary file.")
    
    start = time.perf_counter()  # Start performance measurement
    
    threads = []  # List to store threads
    files = os.listdir()  # Get list of files in directory
    
    # Create threads for reading data files
    for file_name in files:
        if file_name.endswith(".txt") and os.path.isfile(file_name):  # Ensure it's a valid file
            print(f"Reading file: {file_name}")
            thread = threading.Thread(target=read_and_save, args=(file_name, data_path))
            threads.append(thread)
            thread.start()
    
    for thread in threads:
        thread.join()  # Wait for all threads to finish
    
    # Process subjects using multiprocessing
    print("\nStarting multiprocessing to calculate stats...")
    for subject in subjects_lists:
        processes = []  # List to store process instances
        chunk_size = len(subject.scores) // num_cores  # Determine chunk size for each process
        
        # Creating and starting multiple processes for histogram calculation
        for proc in range(num_cores):
            start_idx = proc * chunk_size  # Start index for chunk
            end_idx = len(subject.scores) if proc + 1 == num_cores else (proc + 1) * chunk_size  # End index for chunk
            p = multiprocessing.Process(target=subject.histogram, args=(subject.scores[start_idx:end_idx], queue))  # Creating process
            processes.append(p)  # Adding process to list
            p.start()  # Starting process
        # Waiting for all processes to complete
        for p in processes:
            p.join()
        subject.calculate_stats(num_cores, queue)

    print("\nAll statistics calculated and written to file.")
    
    end = time.perf_counter()  # End performance measurement
    print(f"Time taken: {round(end - start, 3)} seconds")
