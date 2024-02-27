export enum UserRoleTypes {
    PATIENT = 'patient',
    DATASET = 'dataset',
    EDIFIER = 'edifier',
    DOCTOR = 'doctor',
    DATA_SCIENTIST = 'data_scientist',
    ADMIN = 'admin',
    EXTERNAL_SERVER = 'external_server',
    VIEWER = 'viewer'
}

export const isEitherRolesMatch = (rolesToMatch: Array<UserRoleTypes>, userRoles: Array<UserRoleTypes>): boolean => {
    let isMatch = false;
    rolesToMatch.forEach((role) => {
        const index = userRoles.indexOf(role);
        isMatch = index > -1 || isMatch;
    });
    return isMatch;
};

export const isRolesMatch = (rolesToMatch: Array<UserRoleTypes>, userRoles: Array<UserRoleTypes>): boolean => {
    let isMatch = true;
    rolesToMatch.forEach((role) => {
        const index = userRoles.indexOf(role);
        isMatch = index > -1 && isMatch;
    });
    return isMatch;
};
