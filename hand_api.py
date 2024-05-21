import sys
import os
import time
import logging
import math
from math import cos, sin
import numpy as np
import os
import re
import asyncio
import traceback
from rich.traceback import Traceback
from scipy.optimize import root
from hand_model_cal import trans2_theta_psi_D
from hand_cfg import *
import TinyFrame as TF
from CommProtocol import *
import threading


class Hand:
    def __init__(self, 
                 hand_name,
                 target_ip='10.10.20.61',
                 target_port=2000,
                 adjust_time=3, 
                 ):
        self.create_logger()
        if hand_name=="right":
            self.is_right_ = 1
            self.is_left_ = 0
        elif hand_name=="left":
            self.is_right_ = 0
            self.is_left_ = 1
        else:
            self.logger_.error("未指定左右手")
        self.logger_.info("当前使用右手" if self.is_right_ == 1 else "当前使用左手") 
               
        self.target_ip = target_ip
        self.target_port = target_port
        self.adjust_time_ = adjust_time
        self.reader = None
        self.writer = None      
        self.enable_rx = True
        self.rx_cnt = 0  
        self.cmd_ = FingerCommand()
        self.data_ = FingerData()   
        
        self.initialized = 0
        self.thread_ = threading.Thread(target=self.run_loop, daemon=True)
        self.thread_.start()
            
    def create_logger(self):
        self.logger_ = logging.getLogger()
        self.logger_.setLevel(level=logging.INFO)
        handler = logging.FileHandler("freehand.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger_.addHandler(handler)
        self.logger_.addHandler(console)
        self.logger_.info("Start log")
        
    def run_loop(self):
        asyncio.run(self.run())
                
    async def run(self):
        self.logger_.info("初始化...")
        await self.connect()
        self.logger_.info("已连接")
        
        self.tf = TF.TinyFrame()
        self.tf.SOF_BYTE = 0x01
        self.tf.ID_BYTES = 0x01
        self.tf.LEN_BYTES = 0x02
        self.tf.TYPE_BYTES = 0x01
        self.tf.CKSUM_TYPE = 'crc16'
        
        self.tf.write = self.tf
        self.tf.write_async = self.tf_write_async
        self.tf.add_fallback_listener(self.tf_general_listener)
            
        # 电机初始参数   
        # self.enc_dir_   = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1,  1,  0,  0]
        # self.motor_dir_ = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1,  0,  1,  1]
        self.motor_kp_  = [3, 5, 5, 2, 2, 1, 3, 3, 1, 1, 1, 3, 3]
        # self.motor_kp_  = [3, 5, 5, 2, 2, 4, 3, 3, 4, 4, 4, 3, 3]
        self.motor_kd_  = [2, 2, 2, 2, 2, 8, 2, 2, 8, 8, 8, 2, 2]    
        self.motor_dir_ = [-1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1]
        await self.reset_id()
        for i in range(MOTOR_NUM):
            await self.set_motor_ctrl_param(i, 0, 0)
            # await self.set_motor_enc_param(i, self.enc_dir_[i], 1)
            # await self.set_motor_param(i, 0, 2599, self.motor_dir_[i])
        await asyncio.sleep(0.1)
        
        # 运动至零位,其中小拇指和无名指需要上电前复位
        for i in range(MOTOR_NUM):
            if i == 1 or i == 2:
                pass
            else:
                await self.set_motor_target(i, 0, 0, 500*self.motor_dir_[i])
        await asyncio.sleep(0.5)
        for i in range(MOTOR_NUM):
            if i == 1 or i == 2:
                pass
            elif i == 12 or i == 6 or i == 7:
                await self.set_motor_target(i, 0, 0, -3000*self.motor_dir_[i])
            else:
                await self.set_motor_target(i, 0, 0, -1000*self.motor_dir_[i])
        await asyncio.sleep(4)
        self.logger_.info("回到零位")
        
        for i in range(MOTOR_NUM):
            await self.set_motor_target(i, 0, 0, 0)
        self.logger_.info("调整电机...")
        await asyncio.sleep(self.adjust_time_)
        
        # 清零编码器
        for i in range(MOTOR_NUM):
            if i == 1 or i == 2:
                pass
            else:
                await self.clear_enc(i)
        await asyncio.sleep(0.1)
        self.logger_.info("清零编码器")
        
        # if self.writer:
        #     self.writer.close()
        #     await self.writer.wait_closed()
        # exit(1)
        
        # 电机控制初始参数
        for i in range(MOTOR_NUM): 
            await self.set_motor_ctrl_param(i, self.motor_kp_[i], self.motor_kd_[i])
            # await self.set_motor_enc_param(i, self.enc_dir_[i], 1)
        
        # 异步线程-始终获取电机状态
        # await self.reset_id()
        self.enable_rx = 1
        self.recv_task = asyncio.create_task(self.read_forever())
        await self.reset_id()
        await asyncio.sleep(2)
        self.update_task = asyncio.create_task(self.update_data())
        await asyncio.sleep(0.5)
        self.send_task = asyncio.create_task(self.send_cmds())
        await asyncio.sleep(0.5)
        
        self.initialized = 1
        self.logger_.info("初始化完成")
        # await self.recv_task
        
        self.running = 1
        while self.running:
            # print("hand")
            await asyncio.sleep(1)

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.target_ip, self.target_port)
        
    async def send_cmd(self, cmd : CommPack):
        rawdata = cmd.pack()
        await self.tf.send_async(cmd.pkt_type, rawdata)
        # print("s")
        # time.sleep(0.004)
        await asyncio.sleep(0.004)
        
    async def tf_write_async(self, data: bytes) -> int:
        self.writer.write(data)
        await self.writer.drain()
        
    def tf_general_listener(self, tf, frame:TF.TF_Msg):
        self.rx_cnt += 1
        print(f"recv type={frame.type}, len={frame.len}, cnt={self.rx_cnt}")
        if frame.type == COMM_GET_MOTOR_STATE:
            try:
                motor_states_reply = ReplyGetMotorStates(frame.data)
                # os.system("cls")
                # motor_states_reply.print()
                for i, motor_state in enumerate(motor_states_reply.motor_states):
                    id, enc, output = motor_state
                    # print("Motor State {}: id={}, enc={}, output={}".format(i, id, enc, output))
                    
                    # 由于目前版本电机正反转存在问题
                    self.data_.q[id] = enc*self.motor_dir_[id]
                    self.data_.cur[id] = output*self.motor_dir_[id]

                print(f"data: {self.data_.q}")
            except Exception as e:
                tb = traceback.print_exc()
    
    async def reset_id(self):
        pkt = CommIDSet(0)
        await self.send_cmd(pkt)
        
    async def set_motor_ctrl_param(self, motor_id, kp, kd):
        pkt = CommMotorCtrlParam(motor_id, kp, kd)
        await self.send_cmd(pkt)

    async def set_motor_enc_param(self, motor_id, cw_counting_dir, cnt2rad_ratio):
        pkt = CommMotorEncParam(motor_id, cw_counting_dir, cnt2rad_ratio)
        await self.send_cmd(pkt)
        
    async def clear_enc(self, motor_id):
        pkt = CommClearEnc(motor_id)
        await self.send_cmd(pkt)
        
    async def set_motor_param(self, motor_id, min_pwm, max_pwm, cw_dir_level):
        pkt = CommMotorParam(motor_id, min_pwm, max_pwm, cw_dir_level)
        await self.send_cmd(pkt)

    async def set_motor_target(self, motor_id, dest_pos, dest_vel, dest_torque):
        pkt = CommMotorTarget(motor_id, dest_pos, dest_vel, dest_torque)
        await self.send_cmd(pkt)
    
    async def read_forever(self):
        while self.enable_rx:
            data = await self.reader.read(1024)
            self.tf.accept(data)
    
    async def check_alive(self):
        pkt = CommAliveCheck()
        await self.send_cmd(pkt)
    
    async def get_motor_states(self):
        pkt = CommGetMotorStates()
        await self.send_cmd(pkt)
    
    async def update_data(self):
        while True:
            await self.get_motor_states()
            await asyncio.sleep(0.5)
            
    async def send_cmds(self):
        while True:
            for i in range(MOTOR_NUM):
                # TODO: await?
                await self.set_motor_target(i, 
                                            self.cmd_.qDes[i]*self.motor_dir_[i], 
                                            self.cmd_.qdDes[i]*self.motor_dir_[i], 
                                            self.cmd_.tauDes[i]*self.motor_dir_[i])
    
    async def stop(self):
        self.running = 0
        self.thread_.join()
                        
        for i in range(MOTOR_NUM):
            await self.set_motor_ctrl_param(i, 0, 0)
        await asyncio.sleep(0.1)
        
        # 异步事件循环已经结束，不需要手动关闭
        # self.recv_task.cancel()
        # self.update_task.cancel()
        # self.send_task.cancel()
    
        # if self.writer:
        #     self.writer.close()
        #     await self.writer.wait_closed()          
        
    async def close(self):
        await self.stop()
        self.logger_.info("关闭...")  

    def set_param_motors(self, kp=10*np.ones(MOTOR_NUM), kd=0*np.ones(MOTOR_NUM)):
        self.cmd_.kpDes = kp
        self.cmd_.kdDes = kd

    def set_pos_motors(self, pos=np.zeros(MOTOR_NUM)):
        self.cmd_.qDes = pos
        self.cmd_.kpDes = np.array(self.motor_kp_)
        self.cmd_.kdDes = np.array(self.motor_kd_)

    def set_pos_with_param_motors(self, pos=np.zeros(MOTOR_NUM),
                                   kp=10*np.ones(MOTOR_NUM), 
                                   kd=0*np.ones(MOTOR_NUM)):
        self.cmd_.qDes = pos
        self.cmd_.kpDes = kp
        self.cmd_.kdDes = kd
                    
    def set_force_motors(self, tau=np.zeros(MOTOR_NUM)):
        self.cmd_.tauDes = tau
        self.cmd_.kpDes = np.zeros_like(tau)
        self.cmd_.kpDes = np.zeros_like(tau)
        
    async def set_traj_motors(self, pos_end, duration=1, count=1000, method='linear'):
        pass
            
    def map_motor(self, x):
        # 映射电机指令
        motor_map = [8,10,7,5,9,6,2,4,1,3,0,11,12]
        ratio = [12 * 4 * 4,
                 12 * 4 * 4,
                 12 * 256 * 2 * 4,
                 12 * 4 * 4,
                 12 * 4 * 4,
                 12 * 256 * 2 * 4,
                 12 * 256 * 2 * 4,
                 12 * 4 * 4,
                 12 * 256 * 2 * 4,
                 12 * 4 * 4,
                 16 * 90.21 * 8,
                 12 * 16 * 100,
                 12 * 16 * 100]
        x_map = np.zeros_like(x)
        for i in range(MOTOR_NUM):
            x_map[motor_map[i]] = x[i] * ratio[i]
            self.logger_.info(f'{x[i] * ratio[i]}')   
        return x_map

    async def set_pos_joints(self, joints):
        pos = self.map_motor(await self.ik_joints(joints))  # 等待所有逆解完成
        self.set_pos_motors(pos)
  
    async def set_traj_joints(self, joints, duration=0.1, count=2, method='linear'):
        pass
            
    async def ctrl_pos_joints(self):
        # TODO: 目前是关节空间的开环控制，电机只有PD，存在稳态误差
        pass
        
    async def get_data_motors(self):
        # return motor data, TODO:目前只有角度反馈
        motor_data = None
        # TODO
        return motor_data

    async def get_pos_motors(self):
        # 此处是编码器pos
        # TODO
        pass
    
    async def get_pos_joints(self):
        # TODO: 需要正解,存在问题
        motor_pos = None
        with self.lock_:
            motor_pos = deepcopy(self.data_.q)
            
        # motor_data -= self.init_table
        joints = self.fk_joints(motor_pos)
        return joints
    
    def get_module_config(self, name: str):
        return module_config[name]
    
    # 首先是所有四杆机构关系
    def four_bar_forward(self, q, A, B, C, D, degree1_init, degree2_init):
        '''求解四杆机构角度关系,例如从PIP推导DIP
        '''    
        L1=math.sqrt(A**2+C**2-2*A*C*math.cos(q+degree1_init))
        parameter1=(C-A*math.cos(q+degree1_init))/L1
        parameter2=(L1**2+D**2-B**2)/(2*L1*D)
        return degree2_init-(math.acos(parameter2)-math.acos(parameter1))
        
    def four_bar_backward(self, q, A, B, C, D, degree1_init, degree2_init):
        L2=math.sqrt(C**2+D**2-2*C*D*math.cos(degree2_init-q))
        parameter1=math.sin(degree2_init-q)*D/L2
        parameter2=(L2^2+A^2-B^2)/(2*L2*A)
        return math.acos(parameter2)-math.asin(parameter1)-degree1_init
    
    def four_bar_pip2dip(self, q: float, name: str):
        '''所有手指的dip都与pip耦合

        :param q: _description_
        :type q: float
        :param name: _description_
        :type name: str
        :return: _description_
        :rtype: _type_
        '''
        cfg = self.get_module_config(name)
        return self.four_bar_forward(q, cfg["A"],cfg["B"],cfg["C"],cfg["D"],cfg["degree1_init"],cfg["degree2_init"])
        
    def four_bar_dip2pip(self, q: float, name: str):
        cfg = self.get_module_config(name)
        return self.four_bar_backward(q, cfg["A"],cfg["B"],cfg["C"],cfg["D"],cfg["degree1_init"],cfg["degree2_init"])
        
    def four_bar_mcp2pip(self, q: float, name: str):
        assert (name=="fourth" or name=="little")
        cfg = self.get_module_config(name)
        return self.four_bar_forward(q, cfg["Am"],cfg["Bm"],cfg["Cm"],cfg["Dm"],cfg["degree1m_init"],cfg["degree2m_init"])
        
    def four_bar_pip2mcp(self, q: float, name: str):
        assert (name=="fourth" or name=="little")
        cfg = self.get_module_config(name)
        return self.four_bar_backward(q, cfg["Am"],cfg["Bm"],cfg["Cm"],cfg["Dm"],cfg["degree1m_init"],cfg["degree2m_init"])
        
    # ik   
    async def ik_joints(self, q: np.ndarray):
        assert q.shape[0] == 13    
        # 这个地方统一写电机圈数，然后在map里乘对应编码器减速比等    
        d = await asyncio.gather(self.ik_index(q[:3]), 
                      self.ik_middle(q[3:6]), 
                      self.ik_fourth(q[6:8]), 
                      self.ik_little(q[8:10]), 
                      self.ik_thumb(q[10:]))
        self.logger_.info(d)
        d = np.concatenate(d)
        return d

    def ik_driver2(self, q: np.ndarray, name: str):
        '''用于求解无名指与小拇指的逆运动学,其中第一个关节总是指定为直驱关节角度

        :param q: _description_
        :type q: np.ndarray 
        :param name: _description_
        :type name: str
        '''
        assert q.shape[0] == 2
        assert (name=="fourth" or name=="little")
        cfg = self.get_module_config(name)
        psi = q[0]
        theta = q[1]
        bevel = psi / (2*math.pi)
        
        # ts = time.time()
        # rx = np.array([[1, 0, 0],
        #                 [0,cos(psi),-sin(psi)],
        #                 [0,sin(psi),cos(psi)]])    
        # ry = np.array([[cos(theta),0,sin(theta)],
        #                 [0,1,0],
        #                 [-sin(theta),0,cos(theta)]])
        # r = rx @ ry  # 需要交换吗,决定了哪个高哪个低
        # p1r = r @ cfg["p1"]
        # p2r = r @ cfg["p2"]
        
        # def ik_psu(x): # 求根公式太复杂
        #     l2 = np.array([x[0], x[1], x[2]])
        #     p3 = cfg["p3"]
        #     p3[2] = x[3]
        #     l1 = p1r - p3 - l2
        #     equ = l2 - np.dot(l2,p2r)*p2r - np.dot(l1,l2)*l1/cfg["l1"]
            
        #     return [equ[0],
        #             equ[1],
        #             equ[2],
        #             np.linalg.norm(l2)-cfg["l2"]]
    
        # sol_psu = root(ik_psu, np.array([0, 0, 0, 0]), method='lm')
        # self.logger_.debug(f"sol_psu: {sol_psu}")
        # d = sol_psu.x[3]
        # self.logger_.debug(f"d_time: {time.time() -ts}")
        
        # ts = time.time()
        d = trans2_theta_psi_D(theta,psi)/1000
        # self.logger_.debug(f"d2_time: {time.time() -ts}")
        # self.logger_.debug(f"d: {d}")
        # print(d)
        # self.logger_.debug(f"d2: {d2}")
        # self.logger_.debug(f"d_direct: {bevel}")
        
        d = np.array([bevel,d/cfg["h"]])
        return d
        
    def ik_driver3(self, q: np.ndarray, name: str):
        '''用于求解食指,中指的逆运动学,其中第三个关节总是指定为直驱关节角度

        :param q: _description_
        :type q: np.ndarray
        :param name: _description_
        :type name: str
        :return: _description_
        :rtype: _type_
        '''
        assert q.shape[0] == 3
        assert (name=="index" or name=="middle")
        cfg = self.get_module_config(name)
        
        psi = q[0]
        theta = q[1]
        phi = q[2]
        # self.logger_.debug(f"psi: {psi}")
        # self.logger_.debug(f"theta: {theta}")
        # self.logger_.debug(f"phi: {phi}")
        
        # MCP
        rx = np.array([[1, 0, 0],
                        [0,cos(psi),-sin(psi)],
                        [0,sin(psi),cos(psi)]])    
        ry = np.array([[cos(theta),0,sin(theta)],
                        [0,1,0],
                        [-sin(theta),0,cos(theta)]])
        # r = rx @ ry
        r = rx @ ry
        p1r = r @ cfg["p1"]
        p2r = r @ cfg["p2"]
        # self.logger_.debug(f"p1r: {p1r}")
        
        d_left  = -(p1r[2] - math.sqrt(cfg["l1"]**2 - np.sum((p1r[:2]-cfg["p3"][:2])**2)))
        d_right = -(p2r[2] - math.sqrt(cfg["l2"]**2 - np.sum((p2r[:2]-cfg["p4"][:2])**2)))
        self.logger_.debug(f"d_left: {d_left}")
        self.logger_.debug(f"d_right: {d_right}")
        d_left  -= cfg["z0"]
        d_right -= cfg["z0"]
        # print(d_left)
        # d = np.array([d_left, d_right, phi*cfg["bevel"]])
        if self.is_left_:
            d = np.array([d_left/cfg["h"], d_right/cfg["h"], phi/(2*math.pi)])
        else:
            d = np.array([d_right/cfg["h"], d_left/cfg["h"], phi/(2*math.pi)])
        return d
    
    def ik_driver3dc(self, q: np.ndarray, name: str):
        '''用于求解大拇指的逆运动学,一关节为侧摆，二关节为弯曲，第三个关节为直驱关节角度

        :param q: _description_
        :type q: np.ndarray
        :param name: _description_
        :type name: str
        :return: _description_
        :rtype: _type_
        '''
        assert q.shape[0] == 3
        assert (name=="thumb")
        cfg = self.get_module_config(name)
        psi = q[0]
        theta = q[1]
        phi = q[2]
        
        bevel = psi / (2*math.pi)
        # d1是抬升
        d1=cfg["B1"]-cfg["A1"]*math.tan(cfg["degree1_init1"]-math.radians(theta))
        # d2是弯曲
        d2=cfg["B2"]-cfg["A2"]*math.tan(cfg["degree1_init2"]-math.radians(phi))
        d = np.array([bevel, d1/cfg["h"], d2/cfg["h"]])
        return d
        
    async def ik_index(self, q: np.ndarray):
        return self.ik_driver3(q, "index")
    
    async def ik_middle(self, q: np.ndarray): 
        return self.ik_driver3(q, "middle")
    
    async def ik_fourth(self, q: np.ndarray):
        return self.ik_driver2(q, "fourth")
    
    async def ik_little(self, q: np.ndarray):
        return self.ik_driver2(q, "little")
    
    async def ik_thumb(self, q: np.ndarray):
        return self.ik_driver3dc(q, "thumb")
        
    # fk,将给出所有耦合关节的参数
    async def fk_joints(self):
        pass
    
    async def fk_driver2(self, q: np.ndarray, name: str):
        '''用于求解无名指与小拇指的正运动学,其中不包含直驱关节

        :param q: _description_
        :type q: np.ndarray
        :param name: _description_
        :type name: str
        '''
        assert q.shape[0] == 2
        assert (name=="fourth" or name=="little")
        cfg = self.get_module_config(name)
        
       
    async def fk_driver3(self, d: np.ndarray, name: str):
        '''用于求解食指,中指以及大拇指的正运动学,其中d的第三维度总是指定为直驱关节,尽管大拇指的直驱位置与食指不同

        :param d: _description_
        :type d: np.ndarray
        :param name: _description_
        :type name: str
        :return: _description_
        :rtype: _type_
        '''
        assert d.shape[0] == 3
        assert (name=="index" or name=="middle" or name=="thumb")
        cfg = self.get_module_config(name)
        d_left = d[0] # 定义求出的电机位移始终为+,便于发送电机指令,但坐标系中为负数
        d_right = d[1]
        d_direct = d[2]
        # self.logger_.debug(f"d_left: {d_left}")
        # self.logger_.debug(f"d_right: {d_right}")
        
        # pip
        pip = d_direct/cfg["bevel"]
        
        # dip,四杆机构
        L1=math.sqrt(cfg["A"]**2+cfg["C"]**2-2*cfg["A"]*cfg["C"]*math.cos(pip+cfg["degree1_init"]))
        parameter1=(cfg["C"]-cfg["A"]*math.cos(pip+cfg["degree1_init"]))/L1
        parameter2=(L1**2+cfg["D"]**2-cfg["B"]**2)/(2*L1*cfg["D"])
        dip=cfg["degree2_init"]-(math.acos(parameter2)-math.acos(parameter1))
        
        # mcp
        d_left  /= cfg["screw"]
        d_right /= cfg["screw"]
        d_left  += cfg["z0"]
        d_right += cfg["z0"]
        d_left = -d_left # TODO: 有无负号
        d_right = -d_right
        self.logger_.debug(f"d_left: {d_left}")
        self.logger_.debug(f"d_right: {d_right}")
        
        def fk_pss(x):
            rx = np.array([[1, 0, 0],
                            [0,cos(x[0]),-sin(x[0])],
                            [0,sin(x[0]),cos(x[0])]])
            ry = np.array([[cos(x[1]),0,sin(x[1])],
                            [0,1,0],
                            [-sin(x[1]),0,cos(x[1])]])
            r = rx @ ry
            # r = ry @ rx 
            p1r = r @ cfg["p1"]
            p2r = r @ cfg["p2"]
            p3 = cfg["p3"]
            p3[2] = d_left
            p4 = cfg["p4"]
            p4[2] = d_right
            return [np.linalg.norm(r @ p1r - p3) - cfg["l1"],
                    np.linalg.norm(r @ p2r - p4) - cfg["l2"]]
            # return [np.sum((r @ p1r - p3)**2) - cfg["l1"]**2,
            #         np.sum((r @ p2r - p4)**2) - cfg["l2"]**2]
            
        def fk_pss2(x):
            a1 = cfg["p1"][0]
            a2 = cfg["p1"][1]
            a3 = cfg["p1"][2]
            a4 = cfg["p3"][0]
            a5 = cfg["p3"][1]
            a6 = d_left
            b1 = cfg["p2"][0]
            b2 = cfg["p2"][1]
            b3 = cfg["p2"][2]
            b4 = cfg["p4"][0]
            b5 = cfg["p4"][1]
            b6 = d_right
            A1 = np.sum(cfg["p1"]**2) + np.sum(cfg["p3"]**2) - cfg["l1"] ** 2 
            A2 = np.sum(cfg["p2"]**2) + np.sum(cfg["p4"]**2) - cfg["l2"] ** 2 
            
            return [cos(x[1])*a1*a4 + sin(x[1])*a3*a4 + cos(x[0])*a2*a5 + sin(x[0])*a2*a6 + 
                    sin(x[0])*sin(x[1])*a1*a5 - sin(x[0])*cos(x[1])*a3*a5 + cos(x[0])*sin(x[1])*a1*a6 + cos(x[0])*cos(x[1])*a3*a6 - A1 / 2,
                    cos(x[1])*b1*b4 + sin(x[1])*b3*b4 + cos(x[0])*b2*b5 + sin(x[0])*b2*b6 + 
                    sin(x[0])*sin(x[1])*b1*b5 - sin(x[0])*cos(x[1])*b3*b5 + cos(x[0])*sin(x[1])*b1*b6 + cos(x[0])*cos(x[1])*b3*b6 - A2 / 2]
   
        sol_pss = root(fk_pss2, np.array([math.radians(10), math.radians(50)]), method='broyden1')
        # broyden1 hybr anderson (lm) anderson linearmixing diagbroyden excitingmixing
        # print(sol_pss)
        try:
            self.logger_.debug(f"sol_pss: {sol_pss}")
            # self.logger_.debug(f"sol_pss: {type(sol_pss.success)}")
            if sol_pss.success:
                psi = sol_pss.x[0]
                theta = sol_pss.x[1]  # 这句话最后写,urdf才需要减去,后面用解出来的q2???
                q = np.array([psi, theta, pip, dip])
            else:
                q = np.array([None, None, pip, dip])
            return q
        except OSError as reason:
            print(f'reason: {str(reason)}')
                    
    async def fk_index(self, q: np.ndarray):
        return self.fk_driver3(q, "index")
    
    async def fk_middle(self, q: np.ndarray): 
        return self.fk_driver3(q, "middle")
    
    async def fk_fourth(self, q: np.ndarray):
        return self.fk_driver2(q, "fourth")
    
    async def fk_little(self, q: np.ndarray):
        return self.fk_driver2(q, "little")
    
    async def fk_thumb(self, q: np.ndarray):
        return self.fk_driver3(q, "thumb")
    
    async def action(self, action_name, delay=1.5):
        joints = np.zeros(MOTOR_NUM)
        if action_name == "one":
            joints = np.array([0.0,0.4,1.0,0.0,0.4,1.0,0.0,0.4,0.0,0.4,0.0,0.0,0.0])#([0.0,0.3,0.0,0.0,1.7,1.5,0.0,1.5,0.0,1.5,1.2,0.1,0.9])
            # await self.set_pos_joints(joints)
            # time.sleep(0.1)
            # joints = np.array([0,0.3,0,0,1.5,0.9,0,1.5,0,1.5,0,0.7,0.0])
            # self.set_pos_joints(joints)
        elif action_name == "two" or action_name == "yeah":
            joints = np.array([0.3,0.3,0.0,-0.3,0.3,0.0,0.0,1.5,0.0,1.5,1.2,0.1,0.9])
        elif action_name == "three":
            joints = np.array([0.3,0.3,0.0,0.0,0.3,0.0,-0.3,0.0,0.0,1.5,1.2,0.1,0.9])
        elif action_name == "four":
            joints = np.array([0.3,0.3,0.0,0.0,0.3,0.0,0.0,0.0,-0.3,0.0,1.2,0.1,0.9])
        elif action_name == "five":
            joints = np.array([0.3,0.3,0.0,0.0,0.3,0.0,0.0,0.0,-0.3,0.0,0.0,0.0,0.0])
        elif action_name == "six":
            joints = np.array([0.0,1.7,1.5,0.0,1.7,1.5,0.0,1.5,-0.3,0.0,0.0,0.0,0.0])
        elif action_name == "seven":    
            joints = np.array([0.0,1.2,0.5,0.0,1.2,0.5,0.0,1.5,0.0,1.5,1.4,0.1,0.0])
            await self.set_pos_joints(joints)
            await asyncio.sleep(0.5)
            joints = np.array([0.0,1.6,0.4,0.0,1.6,0.4,0.0,1.5,0.0,1.5,1.4,0.1,0.5])
        elif action_name == "eight":    
            joints = np.array([0.0,0.3,0.0,0.0,1.7,1.5,0.0,1.5,0.0,1.5,0.0,0.0,0.0])
        elif action_name == "nine":    
            joints = np.array([0.0,0.3,1.0,0.0,1.7,1.5,0.0,1.5,0.0,1.5,0.0,0.5,1.2])
        elif action_name == "fist":
            joints = np.array([0.0,1.8,0.9,0.0,1.8,0.9,0.0,1.8,0.0,1.8,0.0,0.7,0.5])
        elif action_name == "init":
            joints = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        elif action_name == "open":
            joints = np.array([0.0,0.3,0.0,0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]) 
        elif action_name == "ok":
            joints = np.array([0.0,1.5,1.2,0.0,0.3,0.0,0.0,0.0,-0.3,0.0,1.2,0.0,0.9])
        elif action_name == "tilt":
            joints = np.array([0.3,0.3,0.0,0.3,0.3,0.0,0.3,0.0,0.3,0.0,0.3,0.0,0.0])
        elif action_name == "tilt_reverse":
            joints = np.array([-0.3,0.3,0.0,-0.3,0.3,0.0,-0.3,0.0,-0.3,0.0,-0.3,0.0,0.0])
        elif action_name == "diverge":
            joints = np.array([-0.3,0.3,0.0,-0.3,0.3,0.0,0.3,0.0,0.3,0.0,0,0.0,0.0])
        elif action_name == "pinch1": # 大拇指+食指
            joints = np.array([0.0,1.5,0.7,0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1.0])
        elif action_name == "pinch2":
            joints = np.array([0.0,0.3,0.0,0.0,1.6,0.7,0.0,0.0,0.0,0.0,0.0,1.0,1.3]) 
        elif action_name == "pinch3":
            joints = np.array([0.0,0.3,0.0,0.0,0.3,0.0,0.1,1.1,0.0,0.0,0.0,1.1,1.7]) 
        elif action_name == "pinch4":
            joints = np.array([0.0,0.3,0.0,0.0,0.3,0.0,0.0,0.0,0.0,1.05,0.0,1.1,1.8]) 
        
        await self.set_pos_joints(joints)
        await asyncio.sleep(delay)
        self.logger_.info('='*30)
        self.logger_.info(action_name)
           
async def task(hand:Hand):
    
    # t = time.time()
    # duration = 10
    # test_id = 0
    # while time.time() -t < duration:
    #     # await hand.set_motor_target(test_id, 1800*hand.motor_dir_[test_id],0,0)  # 如果这不await呢
    #     # await asyncio.sleep(0.3)
        
    #     # await hand.set_motor_target(test_id,0*hand.motor_dir_[test_id],0,0)
    #     # await asyncio.sleep(0.3)
        
    #     for i in range(MOTOR_NUM):
    #         if i != 1 and i != 2:
    #             await hand.set_motor_target(i,4000*hand.motor_dir_[i],0,0)     
    #     await asyncio.sleep(0.3)
        
    #     for i in range(MOTOR_NUM):
    #         if i != 1 and i != 2:
    #             await hand.set_motor_target(i,0,0,0)
    #     await asyncio.sleep(0.3)
    
    await hand.action('one')
    await hand.action('two')
    await hand.action('three')
    
    # joints = np.array([0.0,1.57,1.57,0.0,1.57,1.57,0.0,1.57,0.0,1.57,0.0,0.0,0.0])
    # joints = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,1])
    # await hand.set_pos_joints(joints)
    # await asyncio.sleep(0.3)
    
    await hand.close()


if __name__ == '__main__':
    hand = Hand(hand_name="right")
    
    while hand.initialized != 1:
        time.sleep(1)
    
    asyncio.run(task(hand))
    