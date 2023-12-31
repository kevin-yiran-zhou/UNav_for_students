import socket
import threading
import logging
import torch
from os.path import join, exists
from os import makedirs, mkdir
from track import Hloc
from navigation import Trajectory,actions,command_alert,command_normal, command_debug, command_count
import numpy as np
import cv2
import jpysocket
import chardet
from time import time
from datetime import datetime

class Connected_Client(threading.Thread):
    def __init__(self, parent,socket=None, address='128.122.136.173', hloc=None,trajectory=None, connections=None, map_location=None, destinations=None, map_scale=1,log_dir=None, logger=None,initial=False):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = address
        self.id = len(connections)
        self.connections = connections
        self.signal = True
        self.total_connections = 0
        self.hloc = hloc
        self.trajectory=trajectory
        self.map_location=map_location
        self.destination = destinations
        self.destinations_dicts = {}
        for k0, v0 in destinations.items():
            building_dicts = {}
            for k1, v1 in v0.items():
                floor_dicts = {}
                for k2, v2 in v1.items():
                    list0 = []
                    for v3 in v2:
                        list0.append(list(v3.keys())[0])
                    floor_dicts.update({k2: list0})
                building_dicts.update({k1: floor_dicts})
            self.destinations_dicts.update({k0: building_dicts})
        self.log_dir = log_dir
        self.logger = logger
        self.map_scale=map_scale
        self.parent=parent

    def __str__(self):
        return str(self.id) + " " + str(self.address)

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def date(self, s):
        return [s.year, s.month, s.day, s.hour, s.minute, s.second]
    
    def get_alert_distance(self, segment_length, frequency_mode):
        if frequency_mode == "high":
            return max(3, (4*segment_length)**0.5)
        elif frequency_mode == "normal":
            return max(3, (2*segment_length)**0.5)
        else: #low
            return max(3, (0.5*segment_length)**0.5)

    def run(self):
        while self.signal:
            # try:
            self.logger.debug("Waiting to receiver number")
            number = self.recvall(self.socket, 4)
            if not number:
                continue
            command = int.from_bytes(number, 'big')
            instruction_modes = ["overviews", "overview + segments", "segments"]
            frequency_modes = ["high", "normal", "low"]
            if command >= 10 and command < 40: #for app
                self.logger.info('Command: '+str(command))
                #instruction_modes = ["overviews", "overview + segments", "segments"]
                instruction_mode = instruction_modes[command//10 - 1]
                self.logger.info('Instuction mode: '+instruction_mode)
                #frequency_modes = ["high", "normal", "low"]
                frequency_mode = frequency_modes[command%10 - 1]
                self.logger.info('Alert distance '+frequency_mode)
                if instruction_mode == "overview + segments":
                    not_overviewed = True
                self.logger.info('=====Send destination to Client=====')
                destination_dicts = str(self.destinations_dicts) + '\n'
                ###
                # data_bytes = bytes(destination_dicts, 'UTF-8')
                # self.logger.info(destination_dicts)
                # self.socket.sendall(len(data_bytes).to_bytes(4,'big'))
                ###
                self.socket.sendall(bytes(destination_dicts, 'UTF-8'))
            
            elif command >= 40 and command < 70: #for jetson
                self.logger.debug(command)
                # Map the tens digit to the instruction mode
                instruction_index = (command // 10) - 4  
                instruction_mode = instruction_modes[instruction_index]
                self.logger.info('Instruction mode: ' + instruction_mode)
                # Map the ones digit to the frequency mode
                frequency_index = (command % 10) - 1
                frequency_mode = frequency_modes[frequency_index]
                self.logger.info('Alert frequency: ' + frequency_mode)
                if instruction_mode == "overview + segments":
                    not_overviewed = True  # Assuming this is a flag you want to set
                self.logger.debug(frequency_mode+instruction_mode)
            elif command == 3:#for jetson
                self.logger.debug(command)
                self.logger.info('=====Send destination to Client=====')
                destination_dicts = str(self.destinations_dicts) + '\n'
                data_bytes = bytes(destination_dicts, 'UTF-8')
                self.logger.info(destination_dicts)
                self.socket.sendall(len(data_bytes).to_bytes(4,'big'))
                self.socket.sendall(bytes(destination_dicts, 'UTF-8'))
                instruction_mode = instruction_modes[2]
                frequency_mode = frequency_modes[1]
            
            elif command == 2:
                self.logger.debug("Number 2 sent to server")
                return
            
            elif command == 1:
                self.logger.info('===========Loading image===========')
                length = self.recvall(self.socket, 4)
                data = self.recvall(self.socket, int.from_bytes(length, 'big'))
                if not data:
                    continue
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                Destination = self.socket.recv(4096)
                Destination = str(Destination)[7:]
                end_index = Destination.find('\\')
                if end_index == -1:
                    end_index = len(Destination) - 1
                Destination = Destination[:end_index].strip()
                # Destination = jpysocket.jpydecode(Destination)
                Place, Building, Floor, Destination_ = Destination.split(',')
                if Place == self.map_location[0] and Building == self.map_location[1] and Floor == self.map_location[2]:
                    self.logger.info('Destination: '+ Destination + ' (on the same floor)')
                    same_building = True
                    same_floor = True
                    dicts = self.destination[Place][Building][Floor]
                    for i in dicts:
                        for k, v in i.items():
                            if k == Destination_:
                                destination_id = v
                elif Place == self.map_location[0] and Building == self.map_location[1]:
                    self.logger.info('Destination: ' + Destination + ' (in the same building); going to elevator now')
                    same_building = True
                    same_floor = False
                    dicts = self.destination[Place][Building][self.map_location[2]]
                    for i in dicts:
                        for k, v in i.items():
                            if ("elevator" in k) or ("Elevator" in k):
                                destination_id = v
                else:
                    self.logger.warning('Destination not in this building')
                    same_building = False
                    same_floor = False
                    dicts = self.destination[Place][Building][Floor]
                    for i in dicts:
                        for k, v in i.items():
                            if k == Destination_:
                                destination_id = v

                
                self.logger.info('=========Received one image=========')
                pose = self.hloc.get_location(img)
                image_destination = join(
                    self.log_dir, destination_id, 'images')
                if not exists(image_destination):
                    makedirs(image_destination)
                message_destination = join(
                    self.log_dir, destination_id, 'logs')
                if not exists(message_destination):
                    mkdir(message_destination)
                current_time = time()
                readable_date = datetime.fromtimestamp(current_time)
                formatted_date = readable_date.strftime('%Y-%m-%d_%H-%M-%S')

                image_num=len(self.hloc.list_2d)
                cv2.imwrite(
                    join(image_destination, formatted_date+'.png'), img)
                if pose:
                    self.logger.info(f"===============================================\n                                                       Estimated location: x: %d, y: %d, ang: %d\n                                                       Used {image_num} images for localization\n                                                       ===============================================" % (
                        pose[0], pose[1], pose[2]))
                    path_list=self.trajectory.calculate_path(pose[:2], destination_id)
                    if len(path_list) > 0: # have path                        
                        fail_count = 0
                        action_list=actions(pose,path_list,self.map_scale)
                        if instruction_mode == "overviews":
                            message = command_debug(action_list, same_floor)
                        elif instruction_mode == "overview + segments" and not_overviewed: 
                            message = command_debug(action_list, same_floor)
                            not_overviewed = False
                        else: # no overview
                            length = action_list[0][1]
                            if len(action_list) != self.parent.actionlines: # in the first segment or in a new segment
                                self.parent.actionlines = len(action_list)
                                self.parent.halfway = False
                                self.parent.eighty_way = False
                                self.parent.base_len=length
                                self.parent.alert_distance = self.get_alert_distance(length, frequency_mode)
                                if length < self.parent.alert_distance:
                                    message=command_alert(action_list, same_floor)
                                else:
                                    message=command_normal(action_list, same_floor)
                            elif length < self.parent.alert_distance:
                                message=command_alert(action_list, same_floor)
                            else:
                                message=command_count(self.parent,action_list,length, same_floor)
                            message+='\n'
                    else: # no path
                        # message = 'There is no path to the destination. \n'
                        try:
                            fail_count
                        except NameError:
                            fail_count = 1
                        else:
                            fail_count += 1
                        print(f"failed {fail_count} times in a row")
                        if fail_count == 3:
                            message = "Look another direction. \n"
                            fail_count = 0
                        else:
                            message = "\n"
                else: # failed localization
                    try:
                        fail_count
                    except NameError:
                        fail_count = 1
                    else:
                        fail_count += 1
                    print(f"failed {fail_count} times in a row")
                    if fail_count == 3:
                        message = "Look another direction. \n"
                        fail_count = 0
                    else:
                        message = "\n"

                self.logger.info(f"===============================================\n                                                       {message}\n                                                       ===============================================")
                self.socket.sendall(bytes(message, 'UTF-8'))

                with open(join(message_destination, formatted_date+'.txt'), "w") as file:
                    if pose:
                        file.write(str(pose[0])+', '+str(pose[1])+'\n')
                    file.write(message)
            """
            except:
                self.logger.warning("Client " + str(self.address) + " has disconnected")
                self.signal = False
                self.connections.remove(self)
                break
            """

class Server():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    def __init__(self, root, map_data, hloc_config, server_config):
        location_config = server_config['location']
        self.log_path = join(server_config['IO_root'], 'logs', location_config['place'],
                             location_config['building'], str(location_config['floor']))
        self.scale=server_config['location']['scale']
        self.hloc = Hloc(root, map_data, hloc_config)
        self.trajectory=Trajectory(map_data)

        server_configs = server_config['server']
        host = server_configs['host']
        port = server_configs['port']
        ###
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(5)
        ###
        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.sock.bind((host, port))
        ###
        self.connections = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        self.logger.info(f'Setup server at address:{host} port:{port}')
        self.halfway = False
        self.eighty_way = False
        self.actionlines = -100
        self.alert_distance = 5

    def set_new_connections(self, map_data):
        return threading.Thread(target=self.run, args=(map_data,))

    def run(self, map_data):
        while True:
            for index in range(len(self.connections)):
                if not self.connections[index].is_alive():
                    self.logger.debug(f"Dead thread index: {index}")
            sock, address = self.sock.accept()
            # sock, address = self.sock.recvfrom(1024)
            self.connections.append(
                Connected_Client(parent=self,socket=sock, address=address, hloc=self.hloc,trajectory=self.trajectory, connections=self.connections,
                                 map_location=map_data['map_location'], destinations=map_data['destinations'], map_scale=self.scale, log_dir=self.log_path, logger=self.logger))
            self.connections[len(self.connections) - 1].start()
            self.logger.info("New connection at ID " +
                             str(self.connections[len(self.connections) - 1]))
