#!/usr/bin/python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from random import uniform
from matplotlib import animation


class Ball:

    def __init__(self, x, y, radius):

        self.r = radius

        self.acceleration = np.array([0, 0])

        # self.velocity = np.array([randint(0, self.r),
                                 # randint(0, self.r)])
        self.velocity = np.array([uniform(0, 1),
                                  uniform(0, 1)])
        # self.velocity = np.array([-0.1, -0.1])

        self.position = np.array([x, y])


    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]


    def applyForce(self, force):

        self.acceleration = np.add(self.acceleration, force)

    def _normalize(self, v):

        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


    def update(self):

        self.velocity = np.add(self.velocity, self.acceleration)
        # if np.linalg.norm(self.velocity) > 1:
            # self.velocity = self._normalize(self.velocity)
        self.position = np.add(self.position, self.velocity)
        self.acceleration *= 0


class Pack:

    def __init__(self, radius, list_balls):

        self.iter = 0
        self.list_balls = list_balls
        self.r = radius
        self.list_separate_forces = [np.array([0, 0])] * len(self.list_balls)
        self.list_near_balls = [0] * len(self.list_balls)
        self.wait = True


    def _normalize(self, v):

        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


    def run(self):

        # if self.wait:
            # time.sleep(5)
            # self.wait = False

        self.iter += 1
        print(self.iter)
        for ball in self.list_balls:
            self.checkBorders(ball)
            self.checkBallPositions(ball)
            self.applySeparationForcesToBall(ball)
            print(ball.position)

        print("\n")


    def checkBorders(self, ball):

        d = np.sqrt(ball.x**2 + ball.y**2)

        if d >= self.r - ball.r:

            vr = self._normalize(ball.velocity) * ball.r

            # P1 is collision point between circle and container
            P1x = ball.x + vr[0]
            P1y = ball.y + vr[1]
            P1 = np.array([P1x, P1y])

            # tangent vector
            t_v = np.array([-P1[1], P1[0]])

            # Normal vector
            n_v = -1 * self._normalize(P1)

            # print(t_v, n_v)

            u = np.dot(ball.velocity, n_v) * n_v
            w = np.subtract(ball.velocity, u)

            ball.velocity = np.subtract(w, u)

            ball.update()

            # if np.linalg.norm(ball.velocity) == 0:
                # print("changing", ball.velocity, ball.position)
                # ball.velocity = np.array([uniform(0, 1), uniform(0, 1)])
                # # P1 is collision point between circle and container
                # P1x = ball.x + vr[0]
                # P1y = ball.y + vr[1]
                # P1 = np.array([P1x, P1y])

                # # tangent vector
                # t_v = np.array([-P1[1], P1[0]])

                # # Normal vector
                # n_v = -1 * self._normalize(P1)

                # # print(t_v, n_v)

                # u = np.dot(ball.velocity, n_v) * n_v
                # w = np.subtract(ball.velocity, u)

                # ball.velocity = np.subtract(w, u)
                # ball.update()
                # list_neighbours = [e for e in self.list_balls if e is not ball]

                # for neighbour in list_neighbours:

                    # d = self._distanceBalls(ball, neighbour)

                    # if d >= (ball.r + neighbour.r):
                        # print("neigh")
                        # if np.linalg.norm(neighbour.velocity):
                            # # neighbour.velocity = np.array([uniform(0, 1), uniform(0, 1)])
                            # neighbour.velocity = ball.velocity

            # import time
            # time.sleep(1)
            # ball.velocity *= -1

        # if (ball.x - ball.r) < - self.r or (ball.x + ball.r) > self.r:
            # ball.velocity[0] *= -1
            # ball.update()
        # if (ball.y - ball.r) < -self.r or (ball.y + ball.r) > self.r:
            # ball.velocity[1] *= -1
            # ball.update()

    def applySeparationBorder(self, ball):

        d = np.sqrt(ball.x**2 + ball.y**2)

        # print(ball.position)
        P1 = self._normalize(ball.position) * self.r

        # print(P1)

        # tangent vector
        t_v = np.array([-P1[1], P1[0]])
        # print(t_v)

        # Normal vector
        n_v = -1 * self._normalize(P1) / (1 / (d - ball.r)**2)

        # print(t_v, n_v)

        # u = np.dot(ball.velocity, n_v) * n_v
        # w = np.subtract(ball.velocity, u)

        # ball.velocity = np.subtract(ball.velocity, n_v)

        # print(n_v)
        # return self._normalize(n_v)
        return n_v

        # ball.update()

            # if np.linalg.norm(ball.velocity) == 0:
                # print("changing", ball.velocity, ball.position)
                # ball.velocity = np.array([uniform(0, 1), uniform(0, 1)])
                # # P1 is collision point between circle and container
                # P1x = ball.x + vr[0]
                # P1y = ball.y + vr[1]
                # P1 = np.array([P1x, P1y])

                # # tangent vector
                # t_v = np.array([-P1[1], P1[0]])

                # # Normal vector
                # n_v = -1 * self._normalize(P1)

                # # print(t_v, n_v)

                # u = np.dot(ball.velocity, n_v) * n_v
                # w = np.subtract(ball.velocity, u)

                # ball.velocity = np.subtract(w, u)
                # ball.update()

        pass


    def checkBallPositions(self, ball):

        # list_neighbours = [e for e in self.list_balls if e is not ball]
        i = self.list_balls.index(ball)

        # for neighbour in list_neighbours:
        # ot a full loop; if we had two full loops, we'd compare every
        # particle to every other particle twice over (and compare each
        # particle to itself)
        for neighbour in self.list_balls[i + 1:]:

            d = self._distanceBalls(ball, neighbour)

            if d < (ball.r + neighbour.r):
                # if np.linalg.norm(neighbour.velocity):
                    # neighbour.velocity = ball.velocity
                return

        ball.velocity[0] = 0
        ball.velocity[1] = 0


    def getSeparationForce(self, c1, c2):

        steer = np.array([0, 0])

        d = self._distanceBalls(c1, c2)

        if d > 0 and d < (c1.r + c2.r):
            diff = np.subtract(c1.position, c2.position)
            # print("diff", diff)
            diff = self._normalize(diff)
            # diff = np.divide(diff, 1 / (d - c1.r - c2.r)**2)
            diff = np.divide(diff, 1 / d**2)
            steer = np.add(steer, diff)

        return steer


    def _distanceBalls(self, c1, c2):

        x1, y1 = c1.x, c1.y
        x2, y2 = c2.x, c2.y

        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return dist


    def applySeparationForcesToBall(self, ball):

        i = self.list_balls.index(ball)

        # list_neighbours = [e for e in self.list_balls if e is not ball]


        # for neighbour in list_neighbours:
        for neighbour in self.list_balls[i + 1:]:
            j = self.list_balls.index(neighbour)
            forceij = self.getSeparationForce(ball, neighbour)
            # print(forceij)

            if np.linalg.norm(forceij) > 0:
                self.list_separate_forces[i] = np.add(self.list_separate_forces[i], forceij)
                self.list_separate_forces[j] = np.subtract(self.list_separate_forces[j], forceij)
                self.list_near_balls[i] += 1
                self.list_near_balls[j] += 1


        sep_border = self.applySeparationBorder(ball)
        self.list_separate_forces[i] = np.add(self.list_separate_forces[i], sep_border)

        if np.linalg.norm(self.list_separate_forces[i]) > 0:
            # self.list_separate_forces[i] = self._normalize(self.list_separate_forces[i])
            self.list_separate_forces[i] = np.subtract(self.list_separate_forces[i], ball.velocity)
            # self.list_separate_forces[i] = np.clip(np.linalg.norm(self.list_separate_forces[i]), a_min=0, a_max=np.array([1]))
        if self.list_near_balls[i] > 0:
            self.list_separate_forces[i] = np.divide(self.list_separate_forces[i], self.list_near_balls[i])


        separation = self.list_separate_forces[i]
        # print(separation)
        ball.applyForce(separation)
        ball.update()


import time
list_balls = list()

for i in range(5):
    # b = Ball(randint(-15, 15), randint(-15, 15), 5)
    b = Ball(0, 0, randint(3, 7))
    list_balls.append(b)


p = Pack(30, list_balls)

fig = plt.figure()

circle = plt.Circle((0, 0), radius=30, fc='none', ec='k')
plt.gca().add_patch(circle)
plt.axis('scaled')
plt.axes().set_xlim(-50, 50)
plt.axes().set_ylim(-50, 50)
# plt.axes().set_xlim(-200, 200)
# plt.axes().set_ylim(-200, 200)


def init():

    fig.clf()
    return (plt.gca(),)


def draw(i):

    patches = []

    p.run()
    fig.clf()
    circle = plt.Circle((0, 0), radius=30, fc='none', ec='k')
    plt.gca().add_patch(circle)
    plt.axis('scaled')
    plt.axes().set_xlim(-50, 50)
    plt.axes().set_ylim(-50, 50)

    for c in list_balls:
        ball = plt.Circle((c.x, c.y), radius=c.r, picker=True, fc='none', ec='k')
        patches.append(plt.gca().add_patch(ball))

    return patches


co = False
anim = animation.FuncAnimation(fig, draw,
                               frames=500, interval=2, blit=True)


# plt.show()


anim.save('line2.gif', dpi=80, writer='imagemagick')


if __name__ == '__main__':
    pass
