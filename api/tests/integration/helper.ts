import moment from 'moment';
import {unitOfTime} from 'moment';

export const delay = async (delay: string) => {
    const period = delay.match(/[^\d.-]/g).join() as unitOfTime.Base;
    const delayMs = moment.duration(
        parseInt(delay), 
        period
    ).asMilliseconds();
    console.log(`Waiting ${delayMs/1000} second(s)...`);
    return new Promise((resolve) => {
        setTimeout(resolve, delayMs);
    });
};
