import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

def actions(current_pose,path_list,scale):
    xc, yc, an = current_pose
    action_list = []
    for p in path_list:
        xn,yn=p
        distance = np.linalg.norm([xn - xc, yn - yc])
        rot = np.arctan2(xn - xc, yn - yc)/ np.pi * 180+180
        rot_ang = (an - rot) % 360
        rot_clock = (round(rot_ang.squeeze().tolist() / 15) / 2) % 12
        if rot_clock < 1:
            rot_clock += 12
        action_list.append([rot_clock, distance*scale])
        xc, yc ,an= xn, yn,rot
    return action_list

def get_direction(rot_clock):
    if rot_clock>=10.5 or rot_clock<1.5:
        direction = "go straight"
    elif 1.5<=rot_clock<4.5:
        direction = "turn right"
    elif 4.5<=rot_clock<7.5:
        direction = "turn around"
    else:
        direction = "turn left"
    return direction

def command_type0(action_list):
    message = ''
    for i, ac in enumerate(action_list):
        rot_clock, distance = ac
        if rot_clock-int(rot_clock)==0.5:
            message += 'Please walk %.1f meters along %d point 5 clock' % (
                distance, int(rot_clock))
        else:
            message += 'Please walk %.1f meters along %d clock' % (
                distance, int(rot_clock))
        if i < len(action_list) - 1:
            message += '. Then '
        else:
            message += '. And you will arrive the destination.\n'
    return message

def command_alert(action_list):
    message = ''
    rot_clock,next_distance=action_list[0]
    direction = get_direction(rot_clock)
    next_station='your destination' if len(action_list)==1 else ''
    if next_station=='your destination' and next_distance<1:
        message='You have arrived your destination'
    else:
        if rot_clock-int(rot_clock)==0.5:
            message += 'Alert!!!!!!! %s and walk %d steps along %d point 5 clock. ' % (
                direction, int(next_distance/0.69), int(rot_clock))
        else:
            message += 'Alert!!!!!!! %s and walk %d steps along %d clock. ' % (
                direction, int(next_distance/0.69), int(rot_clock))
        if next_station=='':
            rot_clock,next_distance=action_list[1]
            direction = get_direction(rot_clock)
            next_station='your destination' if len(action_list)==2 else ''
            if rot_clock-int(rot_clock)==0.5:
                message += 'And then %s and walk %d steps along %d point 5 clock' % (
                    direction, int(next_distance/0.69), int(rot_clock))
            else:
                message += 'And then %s and walk %d steps along %d clock' % (
                    direction, int(next_distance/0.69), int(rot_clock))
            if next_distance<5:
                if next_station=='your destination':
                    message +=' to arrive '+next_station
                else:
                    rot_clock,next_distance=action_list[2]
                    direction = get_direction(rot_clock)
                    if rot_clock-int(rot_clock)==0.5:
                        message += ', followed by %d point 5 clock turn' % (
                            int(rot_clock))
                    else:
                        message += ', followed by %d clock turn' % (
                            int(rot_clock))
            else:
                if next_station=='your destination':
                    message +=' to arrive '+next_station
                else:
                    message +=next_station
        else:
            message +=' to approach '+next_station
    return message

def command_normal(action_list):
    message = ''
    rot_clock,next_distance=action_list[0]
    direction = get_direction(rot_clock)
    next_station='your destination' if len(action_list)==1 else ''
    if rot_clock-int(rot_clock)==0.5:
        message += '%s and walk %d steps along %d point 5 clock' % (
            direction, int(next_distance/0.69), int(rot_clock))
    else:
        message += '%s and walk %d steps along %d clock' % (
            direction, int(next_distance/0.69), int(rot_clock))
    if next_station=='':
        message += next_station
    else:
        message +=' to approach '+ next_station
    return message

def command_type3(current_pose,path_list,scale):
    if current_pose:
        if len(path_list) > 0:
            action_list=actions(current_pose,path_list,scale)
            whole_message = ''
            got_message = False
            for i, ac in enumerate(action_list):
                rot_clock, distance = ac
                direction = get_direction(rot_clock)
                if rot_clock-int(rot_clock)==0.5:
                    # message = 'Please %s and walk %.1f meters along %d point 5 clock.' % (direction, distance, int(rot_clock))
                    message = 'Please %s and walk %d steps along %d point 5 clock.' % (direction, int(distance/0.69), int(rot_clock))
                else:
                    # message = 'Please %s walk %.1f meters along %d clock.' % (direction, distance, int(rot_clock))
                    message = 'Please %s and walk %d steps along %d clock.' % (direction, int(distance/0.69), int(rot_clock))
                if i < len(action_list) - 1:
                    rot_clock_next, distance_next = action_list[i+1]
                    direction_next = get_direction(rot_clock_next)
                    if distance <= 2:
                        message = 'Please %s. ' % (diection_next)
                    else:
                        message += 'Then %s. ' % (direction_next)
                else:
                    message += 'And you will arrive the destination.'
                message += '\n'
                whole_message += message
                if not got_message:
                    result_message = message
                    got_message = True
            print(whole_message)
            return result_message
        else:
            result_message = 'There is no path to the destination. \n'
            print(result_message)
            return result_message
    else:
        result_message = 'Cannot localize. \n'
        print(result_message)
        return result_message