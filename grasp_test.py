import numpy as np
import time
import asyncio
from hand_api import Hand
async def joints_control(hand:Hand, joints):
    while hand.initialized != 1:
        await asyncio.sleep(1)  # 使用 asyncio.sleep 代替 time.sleep
    await hand.set_pos_joints(joints)
    await asyncio.sleep(1.5)
np.set_printoptions(precision=3, suppress=True)
hand = Hand(hand_name="right")
# joints = np.array([0.0,0.4,0.5,0.0,0.4,1.0,0.0,0.4,0.0,0.4,0.0,0.0,0.0])#([0.0,0.3,0.0,0.0,1.7,1.5,0.0,1.5,0.0,1.5,1.2,0.1,0.9])
asyncio.run(joints_control(hand, np.array([0.0,0.4,0.5,0.0,0.4,1.0,0.0,0.4,0.0,0.4,0.0,0.0,0.0])))
asyncio.run(joints_control(hand, np.array([0.0,0.4,1.0,0.0,0.4,1.0,0.0,0.4,0.0,0.4,0.5,0.0,0.0])))
# async def task(hand:Hand):
    
#     dt = 5
#     await hand.action('open',delay=dt)
    
#     # q0 = np.array([0.0,
#     #                0.3,
#     #                0.0,
#     #                0.0,
#     #                0.3,
#     #                0.0,
#     #                0.0,
#     #                0.0,
#     #                0.0,
#     #                0.0,
#     #                0.0,
#     #                0.0,
#     #                0.0])    
#     # q = q0    
#     # # while(1):
#     # #     q[1] += 1.2
#     # #     await hand.set_pos_joints(q)
#     # #     await asyncio.sleep(dt)
#     # #     q[1] -= 1.2
#     # #     await hand.set_pos_joints(q)
#     # #     await asyncio.sleep(dt)
        
#     # for i in [1,2,4,5,7,9,10,11,12]:
#     #     q[i] += 1.2
#     #     await hand.set_pos_joints(q)
#     #     await asyncio.sleep(dt)
#     #     q[i] -= 1.2
#     #     await hand.set_pos_joints(q)
#     #     await asyncio.sleep(dt)
        
#     # await hand.action('open')
#     # # q = np.array([0,0.3,0,0,0.3,0,0,0,0,0,0,0,0.0])  
#     # # # 4.快速弯曲演示
#     # # for _ in range(2):
#     # #     c = [[1,2],[4,5],[7],[9],[11]]
#     # #     for i in range(5):
#     # #         for id in c[i]:
#     # #             q[id] += 1.8
#     # #             await hand.set_joints_pos(q)
#     # #         await asyncio.sleep(0.2)
#     # #     await asyncio.sleep(dt)
#     # #     for i in range(5):
#     # #         for id in c[i]:
#     # #             q[id] -= 1.8
#     # #             await hand.set_joints_pos(q)
#     # #         await asyncio.sleep(0.2)  
#     # #     await asyncio.sleep(dt)
    
    
#     await hand.action('one')
#     # await hand.action('two')
#     # await hand.action('three')
#     # await hand.action('four')
#     # await hand.action('five')
#     # await hand.action('six')
#     # await hand.action('seven')
#     # await hand.action('eight')
#     # await hand.action('nine')
#     # await hand.action('open')
#     # await hand.action('ok',delay=3)
#     # await hand.action('open')
    
#     # await hand.close()


# if __name__ == '__main__':
#     hand = Hand(hand_name="right")
    
#     while hand.initialized != 1:
#         time.sleep(1)
    
#     asyncio.run(task(hand))
