def expert_control(self, omlist, index):
    ############## MODEL-BASED CONTROLLER (Most frequently used dynamics model) ##########
    ######################################################################################
    # For e-puk dynamics
    # Feedback linearization
    # v1: left wheel speed
    # v2: right wheel speed

    K3 = 1  # interaction between i and j

    # sort neighbors by distance

    # need all neighbors, but only diagonal of adjmatrix is 0 so it is okay
    if True:  # not hasattr(self, 'dictDistance'):
        self.dictDistance = dict()
        for j in range(len(self.scene.robots)):
            # if self.scene.adjMatrix[self.index, j] == 0:
            if self.index == j:
                continue
            robot = self.scene.robots[j]  # neighbor
            self.dictDistance[j] = self.xi.distancepTo(robot.xi)
        self.listSortedDistance = sorted(self.dictDistance.items(),
                                         key=operator.itemgetter(1))
    # velocity in transformed space
    vxp = 0
    vyp = 0

    tauix = 0
    tauiy = 0
    # neighbors sorted by distances in descending order

    #### Use angle to get gabriel graph connections
    lsd = self.listSortedDistance
    jList = []
    for i in range(len(lsd)):
        connected = True
        for k in range(len(lsd)):
            if i == k:
                continue
            ri = lsd[i][0]
            rk = lsd[k][0]
            di = np.array([self.xi.xp - self.scene.robots[rk].xi.xp, self.xi.yp - self.scene.robots[rk].xi.yp])
            dj = np.array([self.scene.robots[ri].xi.xp - self.scene.robots[rk].xi.xp,
                           self.scene.robots[ri].xi.yp - self.scene.robots[rk].xi.yp])
            c = np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj))
            angle = np.degrees(np.arccos(c))
            if (angle >= 90 and i != k):
                connected = False
        if (connected):
            jList.append(lsd[i][0])

    for j in jList:
        robot = self.scene.robots[j]
        pijx = self.xi.xp - robot.xi.xp
        pijy = self.xi.yp - robot.xi.yp
        pij0 = self.xi.distancepTo(robot.xi)
        pijd0 = self.desired_distance
        tauij0 = (pij0 - pijd0) / pij0
        tauix += tauij0 * pijx
        tauiy += tauij0 * pijy

    # Achieve and keep formation
    # tauix, tauiy = saturate(tauix, tauiy, dxypMax)
    vxp += -K3 * tauix
    vyp += -K3 * tauiy

    ##### transform speed to wheels
    kk = 1
    theta = self.xi.theta
    M11 = kk * math.sin(theta) + math.cos(theta)
    M12 = -kk * math.cos(theta) + math.sin(theta)
    M21 = -kk * math.sin(theta) + math.cos(theta)
    M22 = kk * math.cos(theta) + math.sin(theta)

    v1 = M11 * vxp + M12 * vyp
    v2 = M21 * vxp + M22 * vyp

    vmax = self.control_vmax  # wheel's max linear speed in m/s
    vmin = self.control_vmin  # wheel's min linear speed in m/s

    # Find the factor for converting linear speed to angular speed
    if math.fabs(v2) >= math.fabs(v1) and math.fabs(v2) > vmax:
        alpha = vmax / math.fabs(v2)
    elif math.fabs(v2) < math.fabs(v1) and math.fabs(v1) > vmax:
        alpha = vmax / math.fabs(v1)
    else:
        alpha = 1
    v1 = alpha * v1
    v2 = alpha * v2
    # if math.fabs(v1)<vmin:
    #     v1=0
    # if math.fabs(v2)<vmin:
    #     v2=0
    return v1, v2