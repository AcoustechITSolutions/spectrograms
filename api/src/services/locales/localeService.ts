import i18n from 'i18n';

export class LocaleService {
    i18Register: any
    constructor(options: i18n.ConfigurationOptions) {
        this.i18Register = {};
        i18n.configure({...options, register: this.i18Register});
    }

    getCurrentLocale(): string {
        return this.i18Register.getLocale();
    }

    getLocales(): string[] {
        return this.i18Register.getLocales();
    }
 
    setLocale(locale: string) {
        if (this.i18Register.getLocales().indexOf(locale) !== -1) {
            this.i18Register.setLocale(locale);
        }
    }

    translate(phrase: string | i18n.TranslateOptions): string {
        return this.i18Register.__(phrase);
    }

    translatePlurals(phrase: string, count:number): string {
        return this.i18Register.__n(phrase, count);
    }
}
