import numpy as np
import taichi as ti
import datetime
import os
from pyevtk.hl import pointsToVTK
import meshio

ti.init(arch=ti.gpu, default_fp=ti.f64)

export = True
err = 1.0e-5
dim = 3
dx = 0.15       # メッシュサイズと同じ位の大きさに
inv_dx = 1 / dx
young = 5.0e8   # Young率
nu = 0.34       # ポアソン比
rho = 4e1       # 密度
gi = ti.Vector([0.0, 0.0, 0.0]) # 重力などの体積力
bottom_z, upper_z = 0.0, 10.0   # 紐の底面と上面のz座標
omega_rote = 20.0              # 角速度

la, mu = young * nu / ((1+nu) * (1-2*nu)) , young / (2 * (1+nu))
sound_s = ti.sqrt((la + 2 * mu) / rho)  
max_number = 250000         # ループ回数
output_span = 1000          # 出力の間隔
dt_max = 0.1 * dx / sound_s
dt = 3.4e-6

print("dt_max", dt_max)
print("dt", dt)     # dtをdt_max以下に設定

bound = 3
grabing = 3
area_start = ti.Vector([-2.0, -2.0, bottom_z - bound * dx])
box_size = ti.Vector([4.0, 4.0, upper_z + 2 * bound * dx])
nx, ny, nz = int(box_size.x * inv_dx) + 1, int(box_size.y * inv_dx) + 1, int(box_size.z * inv_dx) + 1
base_z_bottom = int((bottom_z - area_start.z) * inv_dx - 0.5)



file_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
folder_name = os.path.splitext(os.path.basename(__file__))[0]
dr = './results/' + folder_name+'/{}'.format(file_name)
msh = meshio.read('./mesh_file/FourWire.msh')       # メッシュファイルの読み込み

num_p = msh.points.shape[0]
num_t = msh.cells_dict["tetra"].shape[0]

def get_p_upper():
    p_upper = np.zeros(0, np.int32)
    for p in range(num_p):
        pos_p_z = msh.points[p][2]
        if ti.abs(pos_p_z - upper_z) < err:
            p_upper = np.append(p_upper, p)
            
    return p_upper

p_upper_np = get_p_upper()
num_p_upper = p_upper_np.shape[0]
p_upper = ti.field(dtype=ti.i32, shape=num_p_upper)
p_upper.from_numpy(p_upper_np)
print("num_p_upper", p_upper)


@ti.data_oriented
class elas_body():
    def __init__(self, msh):
        self.ELE = "tetra"
        self.num_p = msh.points.shape[0]
        self.num_t = msh.cells_dict[self.ELE].shape[0]
        self.m_p = ti.field(dtype=float, shape=self.num_p)
        self.pos_p = ti.Vector.field(dim, dtype=float, shape=self.num_p, needs_grad=True)
        self.pos_p_rest = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.vel_p = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.f_p_ext = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.C_p = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p)
        self.p_I = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))
        self.m_I = ti.field(dtype=float, shape=(nx, ny, nz))
        self.total_energy = ti.field(dtype=float, shape=(), needs_grad=True)
        self.tN_pN = ti.field(dtype=ti.i32, shape=(self.num_t, 4))
        self.vol_t = ti.field(dtype=float, shape=self.num_t)
        self.pos_p.from_numpy(msh.points)
        self.pos_p_rest.from_numpy(msh.points)
        self.tN_pN.from_numpy(msh.cells_dict[self.ELE])
        self.base_z_upper = ti.field(dtype=ti.i32, shape=())
        self.ave_z_upper = ti.field(dtype=float, shape=())     
           
    @ti.kernel
    def cal_m_p_f_p_ext(self):
        for t in range(self.num_t):
            a, b, c, d = self.tN_pN[t,0], self.tN_pN[t,1], self.tN_pN[t,2], self.tN_pN[t,3]
            Ref = ti.Matrix.cols([self.pos_p_rest[b] - self.pos_p_rest[a], self.pos_p_rest[c] - self.pos_p_rest[a], self.pos_p_rest[d] - self.pos_p_rest[a]])
            Vol = 1 / 6 * ti.abs(Ref.determinant())
            for _alpha in ti.static(range(4)):
                alpha = self.tN_pN[t, _alpha]
                self.m_p[alpha] += 0.25 * rho * Vol
                self.f_p_ext[alpha] += 0.25 * rho * Vol * gi
        
    
    @ti.kernel
    def compute_total_energy(self):
        for t in range(self.num_t):
            a, b, c, d = self.tN_pN[t, 0], self.tN_pN[t, 1], self.tN_pN[t, 2], self.tN_pN[t, 3]
            Ref = ti.Matrix.cols([self.pos_p_rest[b] - self.pos_p_rest[a], self.pos_p_rest[c] - self.pos_p_rest[a], self.pos_p_rest[d] - self.pos_p_rest[a]])
            Crn = ti.Matrix.cols([self.pos_p[b] - self.pos_p[a], self.pos_p[c] - self.pos_p[a], self.pos_p[d] - self.pos_p[a]])
            F = Crn @ Ref.inverse()
            Vol = 1 / 6 * ti.abs(Ref.determinant())
            I1 = (F @ F.transpose()).trace()
            J = F.determinant()
            element_energy = 0.5 * mu * (I1 - dim) - mu * ti.log(J) + 0.5 * la * ti.log(J)**2
            self.total_energy[None] += element_energy * Vol
            
    @ti.kernel
    def p2g(self):
        for p in range(self.num_p):
            base = ti.cast((self.pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
            fx = (self.pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            f_p_int = - self.pos_p.grad[p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        I = ti.Vector([i, j, k])
                        dist = (float(I) - fx) * dx
                        weight = w[i].x * w[j].y * w[k].z
                        self.m_I[ix, iy, iz] += weight * self.m_p[p]
                        self.p_I[ix, iy, iz] += weight * (self.m_p[p] * (self.vel_p[p] + self.C_p[p] @ dist) + dt * (f_p_int + self.f_p_ext[p]))
                        
    @ti.kernel
    def grid_op(self):
        for ixiy_iz in range(nx * ny * (bound + grabing)):
            ixiy, _iz = ixiy_iz % (nx * ny), ixiy_iz // (nx * ny)
            ix, iy = ixiy % nx, ixiy // nx
            pos_I_x, pos_I_y = dx * ix + area_start.x, dx * iy + area_start.y
            bottom_iz = base_z_bottom + _iz
            upper_iz = self.base_z_upper[None] + _iz - grabing
            self.p_I[ix, iy, bottom_iz] = ti.Vector([0.0, 0.0, 0.0])
            self.p_I[ix, iy, upper_iz].x = self.m_I[ix, iy, upper_iz] * - pos_I_y * omega_rote
            self.p_I[ix, iy, upper_iz].y = self.m_I[ix, iy, upper_iz] * pos_I_x * omega_rote
            
            
                    
    @ti.kernel
    def g2p(self):
        for p in range(self.num_p):
            base = ti.cast((self.pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
            fx = (self.pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_v_p = ti.Vector([0.0, 0.0, 0.0])
            new_C_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        dist = (float(ti.Vector([i, j, k])) - fx) * dx
                        weight = w[i].x * w[j].y * w[k].z
                        vel_this = self.p_I[ix, iy, iz] / self.m_I[ix, iy, iz]
                        vel_this = ti.Vector([0.0,0.0,0.0]) if self.m_I[ix,iy,iz] == 0.0 else vel_this
                        new_C_p += 4 * inv_dx**2 * weight * vel_this.outer_product(dist)
                        new_v_p += weight * vel_this
                        
            self.vel_p[p] = new_v_p
            self.pos_p[p] += dt * self.vel_p[p]
            self.C_p[p] = new_C_p
    
    @ti.kernel
    def cal_base_z_upper(self):
        self.ave_z_upper[None] = 0.0
        for _p in range(num_p_upper):
            p = p_upper[_p]
            self.ave_z_upper[None] += self.pos_p[p].z / num_p_upper
        self.base_z_upper[None] = int((self.ave_z_upper[None] - area_start.z) * inv_dx - 0.5)
        
        
        
    
    def export_vtk(self, file_name):
        points = self.pos_p.to_numpy()
        tetra = self.tN_pN.to_numpy()
        cells=[
            ('tetra',tetra),
        ]
        mesh_=meshio.Mesh(
            points,
            cells,
            point_data={
            },
            cell_data={
            }
        )
        mesh_.write(file_name)
        
body = elas_body(msh=msh)

def main():
    output_times = 0
    if export:
        os.makedirs(dr, exist_ok=True)
    body.cal_m_p_f_p_ext()
    
    sum_m_p = 0.0
    for p in range(msh.points.shape[0]):
        sum_m_p += body.m_p[p]
    print(rho * 0.25**2 * np.pi * 10 * 4)
    print(sum_m_p)
    
    for time_step in range(max_number):
        body.p_I.fill(0)
        body.m_I.fill(0)
        
        body.cal_base_z_upper()
        with ti.Tape(body.total_energy):
            body.compute_total_energy()
        body.p2g()
        body.grid_op()
        body.g2p()
        if time_step % output_span == 0:
            print(time_step)
            if export:
                body.export_vtk(dr+'/mpm{:05d}.vtu'.format(output_times))
                output_times += 1
        
if __name__ == '__main__':
    main()
