import pyimplicitdist
import poselib
import numpy as np
from scipy.spatial.transform import Rotation as R

def colmap2world(tvec, quat):
    r = R.from_quat(quat)
    rmat = r.as_matrix()
    rmat = rmat.transpose()
    rot = R.from_matrix(r.as_matrix().transpose()).as_rotvec()
    return -np.matmul(rmat, tvec).reshape(3), rot

def coarse_pose(p2d, p3d, initial_pp):
    threshold = 6.0
    p2d_center = [x - initial_pp for x in p2d]
    poselib_pose, info = poselib.estimate_1D_radial_absolute_pose(p2d_center, p3d, {"max_reproj_error": threshold})
    p2d_inlier = p2d[info["inliers"]]
    p3d_inlier = p3d[info["inliers"]]
    initial_pose = pyimplicitdist.CameraPose()
    initial_pose.q_vec = poselib_pose.q
    initial_pose.t = poselib_pose.t
    out = pyimplicitdist.pose_refinement_1D_radial(p2d_inlier, p3d_inlier, initial_pose, initial_pp,
                                                    pyimplicitdist.PoseRefinement1DRadialOptions())
    return out, p2d_inlier, p3d_inlier

def pose_refine(out, p2d_inlier, p3d_inlier):
    refined_initial_pose, pp = out['pose'], out['pp']
    cm_opt = pyimplicitdist.CostMatrixOptions()
    refinement_opt = pyimplicitdist.PoseRefinementOptions()
    cost_matrix = pyimplicitdist.build_cost_matrix(p2d_inlier, cm_opt, pp)
    pose = pyimplicitdist.pose_refinement(p2d_inlier, p3d_inlier, cost_matrix, pp, refined_initial_pose,
                                            refinement_opt)
    qvec = pose.q_vec
    tvec = pose.t
    qvec = [qvec[1], qvec[2], qvec[3], qvec[0]]
    tvec, qvec = colmap2world(tvec, qvec)
    return tvec, qvec

def pose_multi_refine(list_2d, list_3d, initial_poses, pps, rot_base, T, simple_pose):
    cm_opt = pyimplicitdist.CostMatrixOptions()
    refinement_opt = pyimplicitdist.PoseRefinementOptions()
    invalid_id, list_2d_valid, list_3d_valid, initial_poses_valid, pps_valid = [], [], [], [], []
    for i in range(len(list_2d)):
        if isinstance(pps[i], str):
            invalid_id.append(i)
        else:
            list_2d_valid.append(list_2d[i])
            list_3d_valid.append(list_3d[i])
            initial_poses_valid.append(initial_poses[i])
            pps_valid.append(pps[i])
    cost_matrix = pyimplicitdist.build_cost_matrix_multi(list_2d_valid, cm_opt, np.average(pps_valid, 0))
    poses_valid = pyimplicitdist.pose_refinement_multi(list_2d_valid, list_3d_valid, cost_matrix,
                                                        np.average(pps_valid, 0), initial_poses_valid,
                                                        refinement_opt)
    invalid_id=set(invalid_id)
    qvecs = []
    tvecs = []
    j = 0
    for i in range(len(list_2d)):
        if i not in invalid_id:
            qvec = poses_valid[j].q_vec
            tvec = poses_valid[j].t
            qvec = [qvec[1], qvec[2], qvec[3], qvec[0]]
            tvec, qvec = colmap2world(tvec, qvec)
            qvecs.append(qvec)
            tvecs.append(tvec)
            j += 1
        else:
            qvecs.append('None')
            tvecs.append('None')
    tvec, qvec = tvecs[-1], qvecs[-1]
    if simple_pose:
        x_, _, y_ = tvec
        ang = ((-qvec[1] - rot_base)* 180 / np.pi)%360
        tvec = T @ np.array([[x_], [y_], [1]])
        x_, y_ = tvec.tolist()
        return [x_[0],y_[0],ang]
    else:
        """
        qvec is a rotation vector determined according (roughly, for accuracy use -rot_base rotation around vertical axis)
        to basis x = -y_f, y = -z_f, z = -x_f (with respect to floorplan basis)
        [[0, 0, -1]
         [-1, 0, 0]
         [0, -1, 0]]
        We need to transform qvec into the floorplan's basis according to this knowledge
        """
        y_flip = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]])
        y_down = R.from_rotvec([np.pi / 2, 0, 0]).as_matrix() @ y_flip
        rotate_to_world = R.from_rotvec([0, 0, rot_base]).as_matrix() @ y_down
        world_pose = (rotate_to_world @ qvec.reshape(3,1)).reshape(3)
        # Transform coordinates to floorplan bases
        x_pos, z_pos, y_pos = tvec
        world_coor = np.append(T @ np.array([[x_pos], [y_pos], [1]]), [[z_pos]], 0)
        return np.append(world_coor.transpose(), world_pose).tolist()
    

#     public class Main {
#     public static void main(String[] args) {
#         // Initialize the Kalman Filter
#         double initialX = 0.0;
#         double initialP = 1.0;
#         double processNoise = 0.01;
#         double measurementNoise = 0.1;
#         KalmanFilter kalmanFilter = new KalmanFilter(initialX, initialP, processNoise, measurementNoise);

#         // Simulate measurements and filter updates
#         double[] measurements = {     };      // original is an exampke{1.1, 1.2, 1.3, 1.4}
#         for (double measurement : measurements) {
#             // Predict the next state
#             kalmanFilter.predict();

#             // Update the state based on the measurement
#             double filteredValue = kalmanFilter.update(measurement);

#             System.out.println("Measurement: " + measurement + " Filtered Value: " + filteredValue);
#         }
#     }
# }