import pygame
import numpy as np
from abc import ABC, abstractmethod
import random as rn
import scipy.integrate as sci
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0,255,0)
YELLOW = (255,255,0)
RED =  (255,0,0)
GREY = (50,50,50)
GCONST = np.float64(6.64e-11)
def get_maginitude(v):
    result = np.sqrt(get_sqr_distance(v,np.arange(len(v))*0))
    if result < 1:
        return 0
    return  result

def get_sqr_distance(v1, v2):
    result = 0
    for i in range(min(len(v1),len(v2))):
        result += (v1[i] - v2[i])**2
    return  result



def get_unit_vector(v):
    return v / np.sqrt(get_sqr_distance(np.array([0, 0, 0]), v))


class Window:
    isRunning = True
    width, height = 1280, 720
    screens = []
    def __init__(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 25)
        self.add_to_stack(WorldScreen(self))
        deltaTime = 0
        clock = pygame.time.Clock()
        while self.isRunning:
            deltaTime = clock.tick(60) / 1000
            self.get_top_screen().input(pygame.event.get(),deltaTime)
            screen.fill(BLACK)
            self.get_top_screen().update(deltaTime)
            self.get_top_screen().draw(deltaTime, screen)
            screen.blit(self.font.render("FPS:" + "%.3G" % (1/deltaTime) ,True,WHITE),(self.width - 75,self.height -25))

            pygame.display.flip()

    def add_to_stack(self, screen):

        self.screens.append(screen)

    def delete_top_stack(self):
        self.pop_stack()

    def pop_stack(self):
        if len(self.screens) < 1:
            return
        return self.screens.pop()

    def get_top_screen(self):
        return self.screens[-1]

    def switch_screen(self, screen):
        self.delete_top_stack()
        self.add_to_stack(screen)


class Screen(ABC):
    window = None

    @abstractmethod
    def input(self, event,delta):  # take user input here
        for event in event:
            if event.type == pygame.QUIT:
                self.window.isRunning = False

    @abstractmethod
    def draw(self, delta, screen):  # draw output here
        pass

    @abstractmethod
    def update(self, delta):  # do any animations or other pre-draw calculations here - may not be needed
        pass


def randomColour():
    return [rn.randint(0, 255) for a in range(3)]


class WorldScreen(Screen):
    bodies = []
    zoomSpeed = 0.9
    scrollSpeed = 100
    selectedbody = None
    lock = False
    def __init__(self, window):
        self.window = window
        sun = PhysicsBody(1.989e30, [0, 0, 0], [0, 0, 0],6.957e8,'Sun',YELLOW)
        mercury = PhysicsBody(3.30e23,[0,5.79e10,0],[-4.74e4,0,0],2.439e6,'Mercury',(139,69,19),sun)
        venus = PhysicsBody(4.87e24,[0,1.082e11,0],[-3.50e4,0,0],6.052e6,'Venus',(255,233,0),sun)
        earth = PhysicsBody(5.972e24,[1.496e+11,0,0],[0,2.978e4,0],6.371e6,'Earth',(50,50,255),sun)
        mars = PhysicsBody(6.24e23,[2.279e11,0,0],[0,2.41e4,0],3.396e6,'Mars',(255,0,0),sun)
        self.bodies = [sun,
                       mercury,
                       venus,
                       earth,
                       mars]
        self.cam = Camera([0, 0, 0], 6e-9)

    def get_body_under_cursor(self,):
        for body in self.bodies:
            distance = get_sqr_distance(pygame.mouse.get_pos(), self.cam.transformCoords(body.getPosition())[:-1])
            if distance <= (np.log(body.radius)) ** 2:
                return body

    def input(self, events,delta):
        Screen.input(self, events,delta)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.selectedbody = self.get_body_under_cursor()
                elif event.button == 5:
                    self.cam.position += 1 / 2 * np.array(
                        [self.window.width * (1/self.cam.scale) * (1 - 1/self.zoomSpeed), self.window.height * (1/self.cam.scale) * (1 - 1/self.zoomSpeed), 0])
                    self.cam.scale *= self.zoomSpeed
                elif event.button == 4:
                    self.cam.position -= 1 / 2 * np.array([self.window.width * (1/self.cam.scale) * (1 - 1/self.zoomSpeed),
                                                           self.window.height * (1/self.cam.scale) * (1 - 1/self.zoomSpeed), 0])
                    self.cam.scale /= self.zoomSpeed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.lock = True
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            self.cam.position[0] -= 70*delta * 1 / self.cam.scale
            self.lock = False
        if key[pygame.K_RIGHT]:
            self.cam.position[0] += 70 * delta * 1 / self.cam.scale
            self.lock = False
        if key[pygame.K_UP]:
            self.cam.position[1] -= 70 * delta * 1 / self.cam.scale
            self.lock = False
        if key[pygame.K_DOWN]:
            self.cam.position[1] += 70 * delta * 1 / self.cam.scale
            self.lock = False
            #Poll Select

    def draw(self, delta, screen):
        for body in self.bodies:
            pos = [int(a) for a in self.cam.transformCoords(body.getPosition())[:-1]]
            pygame.draw.lines(screen,WHITE,False,[self.cam.transformCoords(a) for a in body.history])
            pygame.draw.circle(screen, body.colour, pos, int(np.log(body.radius)))
        if self.selectedbody is not None:
            nameText = self.window.font.render("Name: " + self.selectedbody.name,True,WHITE)
            velocity = self.window.font.render("Velocity: " + "%.3G" % get_maginitude(self.selectedbody.velocity) + "m/s",True,WHITE)
            screen.blit(nameText,(20,20))
            screen.blit(velocity,(20,40))
            if self.selectedbody.parent is not None:
                nameText = self.window.font.render("Name: " + self.selectedbody.parent.name,True,GREEN)
                velocity = self.window.font.render("Velocity: " + "%.3G" % get_maginitude(self.selectedbody.parent.velocity) + "m/s",True,GREEN)
                relativeVelocity = self.window.font.render("Relative Velocity: " + "%.3G" % get_maginitude(self.selectedbody.velocity - self.selectedbody.parent.velocity) + "m/s",True,WHITE)
                eccentricity = self.window.font.render("Eccentricity: " + "%.3G" % self.get_eccentricity(self.selectedbody,self.selectedbody.parent),True,WHITE)
                sma = self.window.font.render("Semi-Major Axis: " + "%.3G" % self.get_sma(self.selectedbody,self.selectedbody.parent) + "m",True,WHITE)
                screen.blit(nameText,(self.window.width - 20 - nameText.get_size()[0],20))
                screen.blit(velocity,(self.window.width - 20 - velocity.get_size()[0],40))
                screen.blit(relativeVelocity,(20,60))
                screen.blit(eccentricity,(20,80))
                screen.blit(sma,(20,100))

    def get_sma(self,a,b):
        return -1 * GCONST*a.mass*b.mass/(2*self.getOrbitalEnergy(a,b))

    def get_eccentricity(self,a,b):
        val = 1 + (2*self.getOrbitalEnergy(a,b)*(a.mass*get_maginitude(a.getPosition()-b.getPosition())*get_maginitude(a.velocity-b.velocity))**2)/(self.getReducedMass(a,b)*(GCONST*a.mass*b.mass)**2)
        if val < 0:
            return 0
        else:
            return np.sqrt(val)
    def getOrbitalEnergy(self,a,b):
        return 0.5 * a.mass * get_sqr_distance([0,0,0],a.velocity-b.velocity) - GCONST*a.mass*b.mass/get_maginitude(a.getPosition()-b.getPosition())

    def getReducedMass(self,a,b):
        return (a.mass * b.mass)/(a.mass + b.mass)

    def update(self, delta):  # step each body
        delta *= 1e6
        #if self.lock:
        #    self.cam.position = self.selectedbody.getPosition() - self.cam.inverseCoords(np.array([self.window.width/2,self.window.height/2,0]))
        for body in self.bodies:
            a = self.rk4step(body.getPosition(),body.velocity,0.0,(0,0),body)
            b = self.rk4step(body.getPosition(),body.velocity,delta*0.5,a,body)
            c = self.rk4step(body.getPosition(),body.velocity,delta*0.5,b,body)
            d = self.rk4step(body.getPosition(),body.velocity,delta,c,body)

            dxdt = (a[0] +(b[0] + c[0])*2 +d[0])*(1/6)
            dvdt = (a[1] + (b[1] + c[1])*2 + d[1])*(1/6)

            body.setPosition(body.getPosition() + dxdt*delta)
            body.velocity += dvdt * delta

    def rk4step(self,pos,vel,dt,dxdv,body):
        ppos = pos + dxdv[0]*dt
        pvel = vel + dxdv[1]*dt
        return pvel, self.get_accel(ppos, body)

    def get_accel(self,pos,body):
        accel = np.array([0,0,0],dtype=np.float64)
        for other in self.bodies:
            if other is body:
                continue
            mag = ((GCONST * other.mass) / get_sqr_distance(pos,other.getPosition()))
            accel += mag* get_unit_vector(other.getPosition()-pos)
        return accel
class Camera:
    def __init__(self, position, scale):
        self.position = np.array(position, dtype=np.float64)
        self.scale = scale

    def transformCoords(self, coord):
        return (coord - self.position[:len(coord)]) * self.scale

    def inverseCoords(self,coord):
        return (coord/self.scale) + self.position

class PhysicsBody:
    name = "Default"
    _position = []
    velocity = []
    acceleration = []
    def __init__(self, mass, position, init_velocity,radius,name = 'Default',colour = None,parent = None):
        self.mass = mass
        self._position = np.array(position, dtype=np.float64)
        if parent is not None:
            self._position += parent.getPosition()
        self.history = [self._position[:-1]]
        self.velocity = np.array(init_velocity, dtype=np.float64)
        if parent is not None:
            self.velocity += parent.velocity
        self.acceleration = np.array([0, 0, 0], dtype=np.float64)
        self.parent = parent
        self.radius = radius
        if colour is None:
            self.colour = randomColour()
        else:
            self.colour = colour
        self.name = name

    def setPosition(self,newPosition):
        self.history.insert(0,self._position[:-1])
        if len(self.history) > 800:
            self.history.pop()
        self._position = newPosition

    def getPosition(self):
        return self._position
w = Window()
