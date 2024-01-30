import taichi as ti
import numpy as np
import datetime
import meshio
import os
import sys
from math import *
from pyevtk.hl import *

sys.path.append("../my_module/")

from SOLID_P2Q import SOLID_P2Q

ti.init()

USER = "HASHIGUCHI"

USING_MACHINE = "CERVO"
# USING_MACHINE = "GILES"
# USING_MACHINE = "MAC"
EXPORT = True
WEAK_S = True
ATTENUATION_S = True
SOLID_MODEL = "NeoHook"

if USER == "HASHIGUCHI" :
    if USING_MACHINE == "CERVO" :
        ti.init(arch=ti.gpu, default_fp=ti.float64)
        dir_mesh = "./mesh_file"
        FOLDER_NAME = os.path.splitext(os.path.basename(__file__))[0]
        DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE
        dir_vtu = dir_export + "/" + "vtu"
        dir_numpy = dir_export + "/" + "numpy"
        if EXPORT : 
            os.makedirs(dir_vtu, exist_ok=True)
            os.makedirs(dir_numpy, exist_ok=True)

    elif USING_MACHINE == "MAC" :
        ti.init(arch=ti.cpu, default_fp=ti.float64)
        dir_mesh = "./mesh_file"
        if EXPORT : 
            FOLDER_NAME = "decompression3D"
            DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE
            dir_numpy = dir_export + "/" + "numpy"
            dir_vtu = dir_export + "/" + "vtu"
            os.makedirs(dir_numpy, exist_ok=True)
            os.makedirs(dir_vtu, exist_ok=True)
            
    elif USING_MACHINE == "GILES" :
        ti.init(arch=ti.gpu, default_fp=ti.float64)
        FOLDER_NAME = os.path.splitext(os.path.basename(__file__))[0]
        DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        dir_mesh = '/home/hashiguchi/mpm_simulation/geometry/toYuji'
        dir_export = '/home/hashiguchi/mpm_simulation/result/toYuji' + FOLDER_NAME + "/" + DATE
        dir_vtu = dir_export + "/" + "vtu"
        dir_numpy = dir_export + "/" + "numpy"
        if EXPORT :
            os.makedirs(dir_export, exist_ok=True)
            os.makedirs(dir_vtu,  exist_ok=True)
            os.makedirs(dir_numpy, exist_ok=True)
            
            
mesh_name = "stringP2QSize0p05"
Radius = 0.1
Height = 10.0

msh_s = meshio.read(dir_mesh + "/" + mesh_name + ".msh")
dx_mesh = 0.05
num_wire = 4
pos_centerXY_wire = ti.Matrix([
    [2.0 * Radius, 0.0],
    [0.0, 2.0 * Radius],
    [- 2.0 * Radius, 0.0],
    [0.0, - 2.0 * Radius],
])

area_start = ti.Vector([-10.0 * Radius, -10.0 * Radius, - 0.2 * Height])
area_end = ti.Vector([10.0 * Radius, 10.0 * Radius, 1.2 * Height])

err = 1.0e-20
dim = 3
nip = 3
sip = nip
dx = dx_mesh
inv_dx = 1 / dx
rho_s = 1.0e3       # 密度
nu_s = 0.34       # ポアソン比
young_s = 2.0e7   # Young率
alpha_Ti = 0.0 # Young率と引っ張り応力の比
Ti = alpha_Ti * young_s   # 引っ張り応力
gi = ti.Vector([0.0, 0.0, 0.0]) # 重力などの体積力
Z_BOTTOM, Z_UPPER = 0.0, Height   # 紐の底面と上面のz座標(参照状態)

la_s, mu_s = young_s * nu_s / ((1+nu_s) * (1-2*nu_s)) , young_s / (2 * (1+nu_s))
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
dt_max = 0.1 * dx / sound_s
dt = 2.84e-5

print("dt_max", dt_max)
print("dt", dt)


max_number = 200000
output_span = 200

OMEGA = 20.0



@ti.data_oriented
class twistWire(SOLID_P2Q):
    def __init__(self) :
        SOLID_P2Q.__init__(
            self,
            msh_s= msh_s,
            rho_s= rho_s,
            young0_s= young_s,
            nu_s= nu_s,
            la0_s= la_s,
            mu0_s= mu_s,
            dt= dt,
            nip= nip,
            sip= sip,
            WEAK_S= WEAK_S,
            ATTENUATION_S= ATTENUATION_S,
        )
        
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.time_step = ti.field(dtype=ti.i32, shape=())
        self.p_z_add = ti.field(dtype=float, shape=())
        self.m_add = ti.field(dtype=float, shape=())
        
        self.set_taichi_field_wire()
        SOLID_P2Q.set_taichi_field(self)
        SOLID_P2Q.set_aN_a1a2a3(self)
        SOLID_P2Q.set_bN_b1b2(self)
        SOLID_P2Q.leg_weights_roots(self)
        SOLID_P2Q.leg_weights_roots_MPM(self)
        
        self.set_taichi_field_MPM()
        self.set_s_init()
        self.set_p_fix()
        self.set_p_add()
        self.cal_Ja_g()
        self.cal_m_p_s()
        self.cal_m_p_s_MPM()
        self.select_cal_f_int_p_s()
        
        print(self.m_p_s)
        print(self.m_p_s.to_numpy().sum())
        print(self.m_p_s_MPM.to_numpy().sum())
        print(self.rho_s * Radius**2 * pi * num_wire * Height)
        
        for p in range(self.num_p_s) :
            print(self.m_p_s_MPM[p])
        
        sys.exit()
        
        
        if EXPORT :
            self.export_calculation_domain(dir_vtu + "/" + "Domain")
    
    def export_calculation_domain(self, dir) :
        if EXPORT:
            pos = np.array([
                [self.area_start.x, self.area_start.y, self.area_start.z], 
                [self.area_end.x, self.area_start.y, self.area_start.z],
                [self.area_end.x, self.area_end.y, self.area_start.z],
                [self.area_start.x, self.area_end.y, self.area_start.z],
                [self.area_start.x, self.area_start.y, self.area_end.z], 
                [self.area_end.x, self.area_start.y, self.area_end.z],
                [self.area_end.x, self.area_end.y, self.area_end.z],
                [self.area_start.x, self.area_end.y, self.area_end.z]
            ])
            pointsToVTK(
                dir,
                # self.dir_export + "/" + "vtu" + "/" + "Domain",
                pos[:, 0].copy(),
                pos[:, 1].copy(),
                pos[:, 2].copy()
            )
    
    def set_p_fix(self) :
        pos_rest_np = self.pos_p_s_rest.to_numpy()
        pN_fix_np = self.pN_fix.to_numpy()
        p_fix_np = np.arange(0, self.num_p_s)[pos_rest_np[:, 2] == 0.0]
        self.num_p_fix = p_fix_np.shape[0]
        self.p_fix = ti.field(dtype=ti.i32, shape=self.num_p_fix)
        self.p_fix.from_numpy(p_fix_np)
        pN_fix_np[p_fix_np] = self.FIX
        self.pN_fix.from_numpy(pN_fix_np)
        
        
    def set_p_add(self) :
        pos_rest_np = self.pos_p_s_rest.to_numpy()
        self.ADD = 2
        pN_add_np = self.pN_fix.to_numpy()
        p_add_np = np.arange(0, self.num_p_s)[pos_rest_np[:, 2] == Height]
        self.num_p_add = p_add_np.shape[0]
        self.p_add = ti.field(dtype=ti.i32, shape=self.num_p_add)
        self.p_add.from_numpy(p_add_np)
        pN_add_np[p_add_np] = self.ADD
        self.pN_fix.from_numpy(pN_add_np)
        
        
    
    def set_taichi_field_MPM(self) :
        self.area_start = area_start
        self.area_end = area_end
        self.box_size = self.area_end - self.area_start
        self.dx = dx
        self.inv_dx = 1 / self.dx
        self.nx = int(self.box_size.x * self.inv_dx + 1)
        self.ny = int(self.box_size.y * self.inv_dx + 1)
        self.nz = int(self.box_size.z * self.inv_dx + 1)
        
        self.m_I = ti.field(dtype=float, shape=(self.nx, self.ny, self.nz))
        self.p_I = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        self.C_p_s = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=(self.num_p_s))
        self.m_p_s_MPM = ti.field(dtype=float, shape=(self.num_p_s))
         
        
    def select_cal_f_int_p_s(self) :
        if SOLID_MODEL == "NeoHook" :
            self.cal_f_int_p_s = self.cal_f_p_int_s_NeoHook
            
        else :
            sys.exit("err : select solid model")
        
        
    def set_taichi_field_wire(self) :
        self.num_p_s_org = self.num_p_s
        self.num_t_s_org = self.num_t_s
        self.num_es_s_org = self.num_es_s
        self.num_p_s *= num_wire
        self.num_t_s *= num_wire
        self.num_es_s *= num_wire
        self.num_gauss *= num_wire
        
    def set_s_init(self) :
        pos_rest_np = np.zeros(shape=(self.num_p_s, self.dim), dtype=np.float64)
        tN_pN_np = np.zeros(shape=(self.num_t_s, self.num_node_ele_s), dtype=np.float64)
        esN_pN_np = np.zeros(shape=(self.num_es_s, self.num_node_sur_s), dtype=np.float64)
        for i in range(num_wire) :
            pos_rest_np[(i * self.num_p_s_org):((i + 1) * self.num_p_s_org), 0] = self.msh_s.points[:, 0] + pos_centerXY_wire[i, 0]
            pos_rest_np[(i * self.num_p_s_org):((i + 1) * self.num_p_s_org), 1] = self.msh_s.points[:, 1] + pos_centerXY_wire[i, 1]
            pos_rest_np[(i * self.num_p_s_org):((i + 1) * self.num_p_s_org), 2] = self.msh_s.points[:, 2]
            
            tN_pN_np[(i * self.num_t_s_org):((i + 1) * self.num_t_s_org), :] = self.msh_s.cells_dict[self.ELE_s] + i * self.num_p_s_org
            esN_pN_np[(i * self.num_es_s_org):((i + 1) * self.num_es_s_org), :] = self.msh_s.cells_dict[self.SUR_s] + i * self.num_p_s_org
            
        self.pos_p_s.from_numpy(pos_rest_np)
        self.pos_p_s_rest.from_numpy(pos_rest_np)
        self.tN_pN_arr_s.from_numpy(tN_pN_np)
        self.esN_pN_arr_s.from_numpy(esN_pN_np)
        
    def export_SOLID(self, dir):
        cells = [
            (self.ELE_s, self.tN_pN_arr_s.to_numpy())
        ]
        mesh_ = meshio.Mesh(
            self.pos_p_s.to_numpy(),
            cells,
            point_data = {
            },
            cell_data = {
                # "sigma_max" : [sigma_max.to_numpy()],
                # "sigma_mu" : [sigma_mu.to_numpy()],
                # "U_ele" : [U_ele.to_numpy()]
            }
        )
        mesh_.write(dir)
        
    def clear(self) :
        self.f_int_p_s.fill(0)
        self.m_I.fill(0)
        self.p_I.fill(0)
        self.m_add.fill(0)
        self.p_z_add.fill(0)
        
        
    # @ti.kernel
    # def cal_rigid_upper(self) :
    #     for _p in range(self.num_p_add) :
    #         p = self.p_add[_p]
    #         self.m_
        
       
    @ti.kernel    
    def p2g(self) :
        beta = 0.5 * self.dt * self.alpha_Dum[None]
        for p in range(self.num_p_s) :
            # self.vel_p_s[p] = (1 - beta) / (1 + beta) * self.vel_p_s[p] + self.dt * (self.f_int_p_s[p]) / (self.m_p_s[p] * (1 + beta))
            
            
            # if self.pN_fix[p] == self.ADD : 
            #     self.m_add[None] += self.m_p_s[p]
            #     self.p_z_add[None] += self.m_p_s[p] * self.vel_p_s[p].z
                
            base = ti.cast((self.pos_p_s[p] - self.area_start) * self.inv_dx - 0.5, ti.i32)
            fx = (self.pos_p_s[p] - self.area_start) * self.inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)) :
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        I = ti.Vector([i, j, k])
                        dist = (float(I) - fx) * self.dx
                        NpI = w[i].x * w[j].y * w[k].z
                        self.m_I[ix, iy, iz] += NpI * self.m_p_s[p]
                        self.p_I[ix, iy, iz] += NpI * ( (1 - beta) * self.m_p_s[p] * (self.vel_p_s[p] + self.C_p_s[p] @ dist) + self.dt * self.f_int_p_s[p]) / (1 + beta)
                        # self.m_I[ix, iy, iz] += NpI * self.m_p_s_MPM[p]
                        # self.p_I[ix, iy, iz] += NpI * ( (1 - beta) * self.m_p_s_MPM[p] * (self.vel_p_s[p] + self.C_p_s[p] @ dist) + self.dt * self.f_int_p_s[p]) / (1 + beta)
                        # self.p_I[ix, iy, iz] += NpI * ( self.m_p_s_MPM[p] * (self.vel_p_s[p] + self.C_p_s[p] @ dist) )
                        
                        
    @ti.kernel
    def diri_p_I(self) :
        for _p in range(self.num_p_fix) :
            p = self.p_fix[_p]
            base = ti.cast((self.pos_p_s[p] - self.area_start) * self.inv_dx - 0.5, ti.i32)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)) :
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        self.p_I[ix, iy, iz] = [0.0, 0.0, 0.0]     
                        
        for _p in range(self.num_p_add) :
            p = self.p_add[_p]
            base = ti.cast((self.pos_p_s[p] - self.area_start) * self.inv_dx - 0.5, ti.i32)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)) :
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        pos_I_x, pos_I_y = self.dx * ix + self.area_start.x, self.dx * iy + self.area_start.y
                        self.p_I[ix, iy, iz].x = self.m_I[ix, iy, iz] * (- pos_I_y) * OMEGA
                        self.p_I[ix, iy, iz].y = self.m_I[ix, iy, iz] * (pos_I_x) * OMEGA
                        # self.p_I[ix, iy, iz].z = self.p_z_add[None] / self.m_add[None] * self.m_I[ix, iy, iz]
                        self.p_I[ix, iy, iz].z = 0.0
                        
        
    @ti.kernel
    def g2p(self) :
        for p in range(self.num_p_s) :
            base = ti.cast((self.pos_p_s[p] - self.area_start) * self.inv_dx - 0.5, ti.i32)
            fx = (self.pos_p_s[p] - self.area_start) * self.inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_v_p = ti.Vector([0.0, 0.0, 0.0])
            new_C_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)) :
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        I = ti.Vector([i, j, k])
                        dist = (float(I) - fx) * self.dx
                        NpI = w[i].x * w[j].y * w[k].z
                        vel_I = self.p_I[ix, iy, iz] / self.m_I[ix, iy, iz] if self.m_I[ix, iy, iz] > 0.0 else [0.0, 0.0, 0.0]
                        new_v_p += NpI * vel_I
                        new_C_p += 4.0 * self.inv_dx**2 * NpI * vel_I.outer_product(dist)
            self.vel_p_s[p] = new_v_p
            self.C_p_s[p] = new_C_p
            self.pos_p_s[p] += dt * self.vel_p_s[p]
            

            if not(self.vel_p_s[p].z < 1.0e20) :
                self.divergence_s[None] = self.DIVERGENCE
                
    def whether_continue(self):
        if self.divergence_s[None] == self.DIVERGENCE :
            if EXPORT :
                self.export_SOLID()
            sys.exit("error : value was divergenced")
            
    
    @ti.kernel
    def plus_vel_pos_p_s(self):
        beta = 0.5 * self.dt * self.alpha_Dum[None]
        for p in range(self.num_p_s) :
            if self.pN_fix[p] == self.FIX :
                self.vel_p_s[p] = [0.0, 0.0, 0.0]
            elif self.pN_fix[p] == self.ADD :
                self.vel_p_s[p].x = OMEGA * (- self.pos_p_s[p].y)
                self.vel_p_s[p].y = OMEGA * self.pos_p_s[p].x
                self.vel_p_s[p].z = 0.0
                
            elif self.pN_fix[p] == self.FREE :
                self.vel_p_s[p] = (1 - beta) / (1 + beta) * self.vel_p_s[p] + self.dt * (self.f_int_p_s[p]) / (self.m_p_s[p] * (1 + beta))
                
            self.pos_p_s[p] += self.dt * self.vel_p_s[p]
                
            
                
        
    def main(self) :
        while self.time_step[None] <= max_number :
        # while self.time_step[None] <= 0 :
            self.clear()
            self.cal_f_int_p_s()
            self.cal_alpha_Dum()
            self.p2g()
            self.diri_p_I()
            self.g2p()
            
            # self.plus_vel_pos_p_s()
            
            if self.time_step[None] % output_span == 0 :
                print(self.time_step[None])
                if EXPORT :
                    self.export_SOLID(dir_vtu + "/" + "SOLID_{:05d}.vtu".format(self.output_times[None]))
                    self.output_times[None] += 1
                    
            self.whether_continue()
            self.time_step[None] += 1
        
        
if __name__ == "__main__" :
    obj = twistWire()
    obj.main()
