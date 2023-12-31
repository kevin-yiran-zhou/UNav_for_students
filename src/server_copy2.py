import argparse
from os.path import dirname,join,exists,realpath
import yaml
import socket
from conversation import Server,Connected_Client,utils
import loader
import os

def get_server(root,map_data,hloc_config,server_config):
    # IFACE = utils.get_wireless_iface()
    # return Server(server_ip_address=server_config['host'], server_port=server_config['port'], 
    #             feedback_freq=server_config['feedback_freq'], iface=IFACE, fec=server_config['FEC'], debug=server_config['DEBUG'], infer=server_config['INFERENCE'], encoding=server_config['encoding'], sink_type=server_config['sink_type'])
    return Server(root,map_data,hloc_config,server_config)

def main(root,hloc_config,server_config):
    map_data=loader.load_data(server_config)
    server = get_server(root,map_data,hloc_config,server_config)
    newConnectionsThread=server.set_new_connections(map_data)
    newConnectionsThread.start()
    

def receive_file(client_socket, file_name):
    print(f"Receiving file: {file_name}")
    
    # First receive the length of the file.
    fileLengthStr = client_socket.recv(10).decode() # Assuming the length is sent as a string of 10 characters
    fileLength = int(fileLengthStr)
    
    with open(file_name, "wb") as file:
        data = client_socket.recv(1024)
        totalReceived = len(data)
        while totalReceived < fileLength:
            file.write(data)
            data = client_socket.recv(1024)
            totalReceived += len(data)
        
    print(f"File received: {file_name}")



def test():

    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        # Define the port on which you want to connect
        port = 30005
        # Bind to the port
        s.bind(('', port))

        # Allow 1 client to connect
        s.listen(1)
        print("Server listening on port", port)

        while True:
            # Establish connection with client
            c, addr = s.accept()
            print("Got connection from", addr)

            # Get a unique filename for each new connection.
            filename = os.path.join('/home/unav/Desktop/UNav_for_students/src', f'image_{addr[1]}.jpg')

            #/mnt/data/UNav-IO/logs/New_York_University/6_MetroTech/Metrotech_6_Floor_4_With_Stairs/04659/images/2023-06-16_16-05-04.png
            receive_file(c, filename)

            print('Received and saved file', filename)        # Close the connection with the client
            c.close()

    # host = '128.122.136.173'  # Server IP address
    # port = 30001  # Server port number

    # # Create a socket object
    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # # Bind the socket to a specific address and port
    # server_socket.bind((host, port))

    # # Listen for incoming connections
    # server_socket.listen(1)

    # print('Server listening on {}:{}'.format(host, port))

    # # Accept a client connection
    # client_socket, client_address = server_socket.accept()
    # print('Connected to client:', client_address)

    # try:
    #     # Receive data from the client
    #     data = client_socket.recv(1024)
    #     received_message = data.decode('utf-8')
    #     print('Received data:', received_message)

    #     # Send a response back to the client
    #     response = 'Response from Python server: ' + received_message
    #     client_socket.send(response.encode('utf-8'))

    # except Exception as e:
    #     print('Error:', e)

    # finally:
    #     # Close the socket connection
    #     client_socket.close()
    #     server_socket.close()
    #     print('Socket connection closed')

if __name__=='__main__':
    test()
    root = dirname(realpath(__file__)).replace('/src','')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server_config', type=str, default='configs/server.yaml')
    parser.add_argument('-l', '--hloc_config', type=str, default='configs/hloc.yaml')
    args = parser.parse_args()
    with open(args.hloc_config, 'r') as f:
        hloc_config = yaml.safe_load(f)
    with open(args.server_config, 'r') as f:
        server_config = yaml.safe_load(f)
    main(root,hloc_config,server_config)