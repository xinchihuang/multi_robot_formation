import multiprocessing
import random
import signal
import time

def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def writer(queue, filename):
    with open(filename, 'a') as f:
        while True:
            rand_num = queue.get()
            if rand_num == "DONE":
                break
            f.write(str(rand_num) + '\n')
            print(f"Writer wrote {rand_num}")

def generate_randoms(queue):
    ignore_sigint()  # Ignore SIGINT in child process
    while True:
        rand_num = random.randint(1, 100)
        queue.put(rand_num)
        print(f"Process {multiprocessing.current_process().name} generated {rand_num}")
        time.sleep(1)  # Generate a number every second

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    filename = "random_numbers_continuous.txt"

    # Clear the file content
    with open(filename, 'w'):
        pass

    # Start the writer process
    writer_process = multiprocessing.Process(target=writer, args=(queue, filename))
    writer_process.start()

    processes = []
    for _ in range(4):  # Four generator processes
        p = multiprocessing.Process(target=generate_randoms, args=(queue,))
        processes.append(p)
        p.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, terminating processes...")

        # Terminate generator processes
        for p in processes:
            p.terminate()

        # Notify the writer process to terminate
        queue.put("DONE")
        writer_process.join()

    print("All processes completed.")