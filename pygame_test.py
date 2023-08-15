# pygame_test.py

import pygame

# 初始化Pygame
pygame.init()

# 设置窗口尺寸
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("绘制虚线")

# 设置颜色
BLACK = (0, 0, 0)

# 设置虚线参数
dash_length = 10  # 虚线的长度
gap_length = 5  # 虚线之间的间隔

# 绘制虚线函数
def draw_dashed_line(surface, color, start_pos, end_pos, dash_length, gap_length):
    x1, y1 = start_pos
    x2, y2 = end_pos
    # dx = x2 - x1
    # dy = y2 - y1
    # distance = max(abs(dx), abs(dy))
    # dx = dx / distance
    # dy = dy / distance
    # dash_length *= distance
    # gap_length *= distance
    # x, y = x1, y1
    
    pygame.draw.line(surface, color, (x1,y1), (x2,y2), 1)
    # for i in range(int(distance / (dash_length + gap_length))):
    #     pygame.draw.line(surface, color, (round(x), round(y)), (round(x + dx * dash_length), round(y + dy * dash_length)))
    #     x += dx * (dash_length + gap_length)
    #     y += dy * (dash_length + gap_length)

# 游戏主循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 清空屏幕
    screen.fill(BLACK)

    # 绘制虚线
    start_pos = (100, 100)
    end_pos = (700, 500)
    # draw_dashed_line(screen, pygame.Color("white"), start_pos, end_pos, dash_length, gap_length)
    pygame.draw.line(screen, pygame.Color("white"), start_pos, end_pos, 2)

    # 更新屏幕
    pygame.display.flip()

# 退出Pygame
pygame.quit()
