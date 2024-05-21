import struct
from abc import ABC, abstractmethod

COMM_ID_SET = 0xFF
COMM_SET_MOTOR_CTRL_PARAM = 0x01
COMM_SET_MOTOR_PARAM = 0x02
COMM_SET_MOTOR_ENC_PARAM = 0x03
COMM_CLEAR_ENC = 0x04
COMM_SET_MOTOR_TARGET = 0x05

COMM_GET_MOTOR_STATE = 0x80
COMM_ALIVE_CHECK = 0x81

# 定义结构体
class CommPack:
    def __init__(self):
        self.pkt_type = 0xff
        
    @abstractmethod
    def pack(self):
        pass

class CommIDSet(CommPack):
    def __init__(self, id):
        self.min_available_id = id
        self.pkt_type = COMM_ID_SET

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<' + 'B', self.min_available_id)


class CommMotorCtrlParam(CommPack):
    def __init__(self, id, kp, kd):
        self.id = id
        self.kp = kp
        self.kd = kd
        self.pkt_type = COMM_SET_MOTOR_CTRL_PARAM

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<' + 'Bff', self.id, self.kp, self.kd)


class CommMotorParam(CommPack):
    def __init__(self, id, min_pwm, max_pwm, cw_dir_level):
        self.id = id
        self.min_pwm = min_pwm
        self.max_pwm = max_pwm
        self.cw_dir_level = cw_dir_level
        self.pkt_type = COMM_SET_MOTOR_PARAM

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<BHHB', self.id, self.min_pwm, self.max_pwm, self.cw_dir_level)


class CommMotorEncParam(CommPack):
    def __init__(self, id, cw_counting_dir, cnt2rad_ratio):
        self.id = id
        self.cw_counting_dir = cw_counting_dir
        self.cnt2rad_ratio = cnt2rad_ratio
        self.pkt_type = COMM_SET_MOTOR_ENC_PARAM

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<BBf', self.id, self.cw_counting_dir, self.cnt2rad_ratio)


class CommClearEnc(CommPack):
    def __init__(self, id):
        self.id = id
        self.pkt_type = COMM_CLEAR_ENC

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<B', self.id)
    

class CommMotorTarget(CommPack):
    def __init__(self, id, dest_pos, dest_vel, dest_torque):
        self.id = id
        self.dest_pos = dest_pos
        self.dest_vel = dest_vel
        self.dest_torque = dest_torque
        self.pkt_type = COMM_SET_MOTOR_TARGET

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<Bfff', self.id, self.dest_pos, self.dest_vel, self.dest_torque)
    
class CommAliveCheck(CommPack):
    def __init__(self):
        self.id = id
        self.pkt_type = COMM_ALIVE_CHECK

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<B', 0)
    
class CommGetMotorStates(CommPack):
    def __init__(self):
        self.id = id
        self.pkt_type = COMM_GET_MOTOR_STATE

    def pack(self):
        # 将结构体打包为 bytes
        return struct.pack('<B', 0)
    
class ReplyGetMotorStates():
    def __init__(self, bytes):
        # 定义结构体格式
        comm_motor_state_seg_format = "<Bfh"

        # 解析comm_get_motor_state结构体
        self.total_reply, = struct.unpack_from("B", bytes, 0)
        motor_states_bytes = bytes[1:]
        
        self.motor_states = []
        
        for i in range(self.total_reply):
            offset = i * struct.calcsize(comm_motor_state_seg_format)
            motor_state = struct.unpack_from(comm_motor_state_seg_format, motor_states_bytes, offset)
            self.motor_states.append(motor_state)
    
    def print(self):
        # 打印解析结果
        print("Total Reply: {}".format(self.total_reply))
        for i, motor_state in enumerate(self.motor_states):
            id, enc, output = motor_state
            print("Motor State {}: id={}, enc={}, output={}".format(i, id, enc, output))