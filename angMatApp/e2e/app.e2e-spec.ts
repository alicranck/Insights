import { AngMatAppPage } from './app.po';

describe('ang-mat-app App', () => {
  let page: AngMatAppPage;

  beforeEach(() => {
    page = new AngMatAppPage();
  });

  it('should display message saying app works', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('app works!');
  });
});
