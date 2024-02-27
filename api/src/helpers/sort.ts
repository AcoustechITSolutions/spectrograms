import {SortingOrder, SortingParameters} from '../interfaces/SortingParams';

export const createSortByRegExp = (sortingFileds: string[]) => {
    const sortFields = sortingFileds.join('|');
    return new RegExp('(asc|desc)\\((' + sortFields + ')\\)');
};

export const getSortingParamsByFields = (sortByString: string | undefined, sortingFields: string[]): SortingParameters | undefined => {
    const regExp = createSortByRegExp(sortingFields);
    return getSortingParamsByRegExp(sortByString, regExp);
};

export const getSortingParamsByRegExp = (sortByString: string | undefined, sortingRegexp: RegExp): SortingParameters | undefined => {
    if (sortByString == undefined) {
        return undefined;
    }
    const execResult = sortingRegexp.exec(sortByString);
    return handleExecResult(execResult);
};

const handleExecResult = (execResult: RegExpExecArray): SortingParameters | undefined => {
    if (execResult != undefined) {
        return {
            sortingOrder: execResult[1].toUpperCase() as SortingOrder,
            sortingColumn: execResult[2],
        };
    } else {
        return undefined;
    }
};

export const sortByString = (firstString: string, secondString: string, order: SortingOrder) => {
    if (order == 'ASC') {
        return firstString < secondString ? 1 : -1;
    } else {
        return firstString > secondString ? 1 : -1;
    }
};
