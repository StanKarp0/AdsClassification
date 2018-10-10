var save_list = ['Clothing & Shoes',
             'Automotive',
             'Baby',
             'Health & Beauty',
             'Media',
             'Consumer Electronics',
             'Console & Video Games',
             'Tools & Hardware',
             'Outdoor Living',
             'Grocery',
             'Home',
             'Betting',
             'Jewelery & Watches',
             'Stationery & Office Supplies',
             'Pet Supplies',
             'Computer Software',
             'Sports',
             'Toys & Games',
             'Social Sites',
                 'Apartments'];

for (var i = 0; i < save_list.length; i++) {
    console.log(i + 1, save_list[i])
    browser.menus.create({
        id: 'ads-menu-' + (i + 1),
        title: save_list[i],
        contexts: ["image"]
    });
}

browser.menus.onClicked.addListener((info, tab) => {
    var directory = info.menuItemId.substring(9);
    var filename = info.srcUrl.substring(info.srcUrl.lastIndexOf('/')+1);    
    browser.downloads.download({
        url: info.srcUrl,
        filename : 'ads/' + directory + '/' + filename
    });
});
