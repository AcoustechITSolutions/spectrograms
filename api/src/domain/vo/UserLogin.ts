export class UserLogin {
    private login: string
    private constructor(login: string) {
        this.login = login;
    }

    public static create(login: string) {
        const loweredLogin = login.toLowerCase();
        return new UserLogin(loweredLogin);
    }

    public getLogin(): string {
        return this.login;
    }
}
