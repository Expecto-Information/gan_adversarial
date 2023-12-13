import time

class Timer:
    def __init__(self, save_path):
        self.start_time = None
        self.measured_time = {}
        self.save_path = save_path
    def time_measure(self, epoch):
        if self.start_time is None:
            self.start_time = time.time()
        else:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.measured_time[epoch] = elapsed_time
            self.start_time = None
            self.save_info()
    def save_info(self):
         with open(self.save_path, 'w') as file:
             file.write(f'total time is {sum([time for time in self.measured_time.values()])/60} mins\n')
             for epoch, time_value in self.measured_time.items():
                file.write(f'epoch {epoch}: {time_value/60} mins\n')