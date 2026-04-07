import os
import sys
import re


class Logger:
    def __init__(self, filename):

        self.filename = os.path.join(os.getcwd(), filename)
        self.terminal = sys.stdout
        self.reset_logs()
        self.log = open(self.filename, "w")
        self.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

    def reset_logs(self):
        with open(self.filename, 'w') as file:
            file.truncate(0)

    def read_logs(self):
        sys.stdout.flush()

        
        with open(self.filename, "r") as f:#读取全部日志内容
            log_content = f.readlines()

        
        log_content = [line for line in log_content if '\x00' not in line]# 过滤垃圾字符
 
        progress_pattern = re.compile(r'\[.*\] \d+\.\d+%')

        progress_lines = [line for line in log_content if
                          progress_pattern.search(line) and " - Completed!\n" not in line]#找出进度条

        if progress_lines:
            valid_content = [line for line in log_content if line not in progress_lines]
            if log_content[-1] == progress_lines[-1]:
                valid_content.append(progress_lines[-1].strip("\n"))
        else:
            valid_content = log_content

     
        recent_lines = valid_content[-300:]#防止内存过大

  
        return ''.join(recent_lines)
